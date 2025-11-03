#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    NOTICE: This code derived from nnunetv2: https://github.com/MIC-DKFZ/nnUNet

## Title: Fast-nnUNet: High-performance medical image segmentation framework based on the nnUNetv2 architecture
## Authors: Justin Lee
## Description: Knowledge Distillation Based on nnUNet architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU, ConvTranspose3d
from torch import GradScaler
from collections import OrderedDict
import numpy as np
import os
import json
from typing import Union, List, Tuple
from torch import distributed as dist
from torch.cuda import device_count
from time import time
from batchgenerators.utilities.file_and_folder_operations import *
from torch._dynamo import OptimizedModule

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from datetime import datetime
from time import sleep
import sys
try:
    from torch.amp import autocast  # Recommended modern import method
except ImportError:
    try:
        from torch.cuda.amp import autocast
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def autocast(enabled=True, device_type="cuda"):
            yield
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

# Use generic components from dynamic network architecture library
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5

# Lightweight nnU-Net student model consistent with the original nnUNet architecture
class LiteNNUNetStudent(nn.Module):
    """
    Based on the original nnUNet architecture but with reduced feature channels
    """
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 n_stages: int = 6,
                 features_per_stage: list = None,
                 conv_op: type = Conv3d,
                 kernel_sizes: list = None,
                 strides: list = None,
                 n_conv_per_stage: list = None,
                 n_conv_per_stage_decoder: list = None,
                 conv_bias: bool = True,
                 norm_op: type = InstanceNorm3d,
                 norm_op_kwargs: dict = None,
                 dropout_op: type = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: type = LeakyReLU,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = True
                 ):
        super().__init__()
        
        # Parameter settings
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 320, 320]
        if kernel_sizes is None:
            kernel_sizes = [(3, 3, 3)] * n_stages
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * n_stages
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)
        if strides is None:
            strides = [(1, 1, 1)] + [(2, 2, 2)] * (n_stages - 1)
            
        # Check if parameter lengths match
        if not (len(features_per_stage) == n_stages and len(kernel_sizes) == n_stages and 
                len(strides) == n_stages and len(n_conv_per_stage) == n_stages):
            raise ValueError("Encoder parameter list lengths do not match")
        if len(n_conv_per_stage_decoder) != (n_stages - 1):
            raise ValueError("Decoder convolution list length should be n_stages - 1")
            
        # Save initialization parameters
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.n_conv_per_stage = n_conv_per_stage
        self.n_conv_per_stage_decoder = n_conv_per_stage_decoder
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.deep_supervision = deep_supervision
        
        # Build encoder
        self.encoder = PlainConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True
        )
        
        # Build decoder
        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            nonlin_first=False,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

# Lightweight ResEnc student model based on ResidualEncoderUNet architecture
class LiteResEncStudent(nn.Module):
    """
    Based on the ResidualEncoderUNet architecture but with reduced feature channels and blocks
    """
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 n_stages: int = 6,
                 features_per_stage: list = None,
                 conv_op: type = Conv3d,
                 kernel_sizes: Union[int, list] = 3,
                 strides: list = None,
                 n_blocks_per_stage: list = None,
                 n_conv_per_stage_decoder: list = None,
                 conv_bias: bool = True,
                 norm_op: type = InstanceNorm3d,
                 norm_op_kwargs: dict = None,
                 dropout_op: type = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: type = LeakyReLU,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = True
                 ):
        super().__init__()
        
        # Parameter settings
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 320, 320]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if n_blocks_per_stage is None:
            # Reduced from ResEnc's (1, 3, 4, 6, 6, 6) to lighter version
            n_blocks_per_stage = [1, 2, 2, 3, 3, 3][:n_stages]
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [1] * (n_stages - 1)
        if strides is None:
            strides = [(1, 1, 1)] + [(2, 2, 2)] * (n_stages - 1)
            
        # Check if parameter lengths match
        if not (len(features_per_stage) == n_stages and len(kernel_sizes) == n_stages and 
                len(strides) == n_stages and len(n_blocks_per_stage) == n_stages):
            raise ValueError("Parameter list lengths do not match n_stages")
        if len(n_conv_per_stage_decoder) != (n_stages - 1):
            raise ValueError("Decoder convolution list length should be n_stages - 1")
            
        # Save initialization parameters
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.n_blocks_per_stage = n_blocks_per_stage
        self.n_conv_per_stage_decoder = n_conv_per_stage_decoder
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.deep_supervision = deep_supervision
        
        # Use ResidualEncoderUNet directly but with reduced parameters
        self.network = ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        return self.network(x)
    
    @property
    def decoder(self):
        """Provide access to decoder for compatibility with parent class"""
        return self.network.decoder

# Distillation loss
def distillation_loss_fn(student_logits, teacher_logits, temperature):
    """Calculate KL divergence distillation loss"""

    student_logits = student_logits.to(torch.float32)
    teacher_logits = teacher_logits.to(torch.float32)
    
    soft_teacher_logits = teacher_logits.detach() / temperature
    soft_student_logits = student_logits / temperature

    teacher_probs = F.softmax(soft_teacher_logits, dim=1)
    log_student_probs = F.log_softmax(soft_student_logits, dim=1)

    loss = F.kl_div(log_student_probs, teacher_probs, reduction='mean', log_target=False)
    
    scaled_loss = loss * (temperature ** 2)
        
    return scaled_loss

class nnUNetDistillationTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, 
                 teacher_model_folder=None, 
                 teacher_fold=0,
                 teacher_checkpoint_name='checkpoint_final.pth',
                 alpha=0.3,
                 temperature=3.0,
                 feature_reduction_factor=2,
                 block_reduction_strategy='keep',
                 rotate_training_folds=False,
                 rotate_folds_frequency=5,
                 device=torch.device('cuda')):
        """
        nnUNet knowledge distillation trainer
        
        Parameters:
            plans, configuration, fold, dataset_json: Same as regular nnUNetTrainer
            teacher_model_folder: Teacher model folder
            teacher_fold: Teacher model fold number, can be single integer or integer list
            teacher_checkpoint_name: Teacher model checkpoint file name
            alpha: Distillation loss weight
            temperature: Distillation temperature
            feature_reduction_factor: Feature channel reduction factor (2 means channel number halved)
            block_reduction_strategy: Strategy for residual blocks compression
                - 'reduce': Reduce blocks by half (original strategy A)
                - 'keep': Keep original blocks (strategy B) 
                - 'increase': Increase blocks by 1 per stage (strategy B+)
                - 'adaptive': Adaptive increase based on compression ratio (strategy B++)
            rotate_training_folds: Whether to rotate training folds during training
            rotate_folds_frequency: How often to rotate folds (in epochs)
            device: Compute device
        """
        import inspect
        original_init = nnUNetTrainer.__init__
        
        def patched_init(self, plans, configuration, fold, dataset_json, device):

            self.is_ddp = dist.is_available() and dist.is_initialized()
            self.local_rank = 0 if not self.is_ddp else dist.get_rank()
            self.device = device

            if self.is_ddp:
                print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                      f"{dist.get_world_size()}."
                      f"Setting device to {self.device}")
                self.device = torch.device(type='cuda', index=self.local_rank)
            else:
                if self.device.type == 'cuda':
                    self.device = torch.device(type='cuda', index=0)
                print(f"Using device: {self.device}")

            self.my_init_kwargs = {
                'plans': plans,
                'configuration': configuration,
                'fold': fold,
                'dataset_json': dataset_json,
                'device': device
            }
            
            self.plans_manager = PlansManager(plans)
            self.configuration_manager = self.plans_manager.get_configuration(configuration)
            self.configuration_name = configuration
            self.dataset_json = dataset_json
            self.fold = fold

            self.label_manager = self.plans_manager.get_label_manager(dataset_json)

            self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
                if nnUNet_preprocessed is not None else None
            self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                           self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
                if nnUNet_results is not None else None
            self.output_folder = join(self.output_folder_base, f'fold_{fold}')

            self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                    self.configuration_manager.data_identifier)
            self.dataset_class = None
            self.is_cascaded = self.configuration_manager.previous_stage_name is not None
            self.folder_with_segs_from_previous_stage = \
                join(nnUNet_results, self.plans_manager.dataset_name,
                     self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                     self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                    if self.is_cascaded else None

            self.initial_lr = 1e-2
            self.weight_decay = 3e-5
            self.oversample_foreground_percent = 0.33
            self.probabilistic_oversampling = False
            self.num_iterations_per_epoch = 250
            self.num_val_iterations_per_epoch = 50
            self.num_epochs = 1000
            self.current_epoch = 0

            self.enable_deep_supervision = True
            self.allow_val_padding = True

            self.dataloader_train = self.dataloader_val = None

            self._best_ema = None

            self.inference_allowed_mirroring_axes = None

            self.save_every = 50
            self.disable_checkpointing = False

            self.was_initialized = False

            self.grad_scaler = GradScaler("cuda") if device.type == 'cuda' else None
            self.network = None
            self.optimizer = self.lr_scheduler = None  # -> self.initialize
            self.loss = None

            timestamp = datetime.now()
            maybe_mkdir_p(self.output_folder)
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            self.logger = nnUNetLogger()

            self.print_to_log_file("\n#######################################################################\n"
                                   "Please cite the following paper when using nnU-Net:\n"
                                   "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                                   "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                                   "Nature methods, 18(2), 203-211.\n"
                                   "#######################################################################\n",
                                   also_print_to_console=True, add_timestamp=False)
        
        nnUNetTrainer.__init__ = patched_init
        
        try:
            self.teacher_model_folder = teacher_model_folder
            self.teacher_fold = teacher_fold if isinstance(teacher_fold, (list, tuple)) else [teacher_fold]
            self.teacher_checkpoint_name = teacher_checkpoint_name
            self.alpha = alpha
            self.temperature = temperature
            self.feature_reduction_factor = feature_reduction_factor
            self.block_reduction_strategy = block_reduction_strategy
            self.rotate_training_folds = rotate_training_folds
            self.rotate_folds_frequency = rotate_folds_frequency
            self.initial_fold = fold
            self.all_available_folds = None
            self.fold_rotation_counter = 0
            self.student_plans_identifier = 'nnUNetPlans'  # Default
            
            super().__init__(plans, configuration, fold, dataset_json, device)
            
            self.my_init_kwargs.update({
                'teacher_model_folder': teacher_model_folder,
                'teacher_fold': teacher_fold,
                'teacher_checkpoint_name': teacher_checkpoint_name,
                'alpha': alpha,
                'temperature': temperature,
                'feature_reduction_factor': feature_reduction_factor,
                'block_reduction_strategy': block_reduction_strategy,
                'rotate_training_folds': rotate_training_folds,
                'rotate_folds_frequency': rotate_folds_frequency
            })
            
        finally:
            nnUNetTrainer.__init__ = original_init
        
        self.teacher_models = []

    def initialize_fold_rotation(self):
        """Initialize fold rotation by identifying all available folds in the dataset"""
        if not self.rotate_training_folds:
            return
            
        # Determine all available folds based on split file
        split_file = join(self.preprocessed_dataset_folder, "splits_final.json")
        if not isfile(split_file):
            self.print_to_log_file(f"Warning: Cannot find splits_final.json at {split_file}, disabling fold rotation")
            self.rotate_training_folds = False
            return
            
        # Load splits file
        with open(split_file, 'r') as f:
            splits = json.load(f)
            
        # Get number of folds from splits
        self.all_available_folds = list(range(len(splits)))
        
        self.print_to_log_file(f"Fold rotation enabled. Available folds: {self.all_available_folds}")
        self.print_to_log_file(f"Will rotate folds every {self.rotate_folds_frequency} epochs")
        
    def update_fold_for_next_rotation(self):
        """
        Update fold for next training period based on rotation schedule
        Returns true if fold was changed, false otherwise
        """
        if not self.rotate_training_folds or self.all_available_folds is None:
            return False
            
        # Check if it's time to rotate
        if (self.current_epoch % self.rotate_folds_frequency) != 0 or self.current_epoch == 0:
            return False
            
        # Determine next fold
        current_fold_index = self.all_available_folds.index(self.fold)
        next_fold_index = (current_fold_index + 1) % len(self.all_available_folds)
        next_fold = self.all_available_folds[next_fold_index]
        
        # Skip rotation if we've gone through all folds
        if self.fold_rotation_counter >= len(self.all_available_folds):
            self.print_to_log_file(f"Completed all fold rotations, returning to original fold {self.initial_fold}")
            next_fold = self.initial_fold
            self.fold_rotation_counter = 0
        
        # Update fold if different
        if next_fold != self.fold:
            self.print_to_log_file(f"Rotating training fold from {self.fold} to {next_fold}")
            self.fold = next_fold
            self.fold_rotation_counter += 1
            
            # Reinitialize data loaders with new fold
            self.print_to_log_file("Reinitializing data loaders for new fold")
            self.do_split()
            self.setup_DA_params()
            
            # Reinitialize data loaders
            self.dl_tr, self.dl_val = self.get_plain_dataloaders()
            
            # Get batch generators
            self.tr_gen, self.val_gen = self._get_batch_generators()
            
            return True
            
        return False
    
    def initialize(self):
        # Check if already initialized, if so, return directly
        if self.was_initialized:
            self.print_to_log_file("Trainer was already initialized, skipping initialization")
            return
            
        # First execute parent class initialization
        super().initialize()
        
        # Initialize fold rotation if enabled
        self.initialize_fold_rotation()
        
        # Load teacher model
        self.load_teacher_model()
        
        # Initialize log record list
        if self.logger is not None:
            if 'train_seg_losses' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['train_seg_losses'] = []
            if 'train_distill_losses' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['train_distill_losses'] = []
        
        # Output distillation parameters
        self.print_to_log_file(f"Distillation parameters:")
        self.print_to_log_file(f"alpha (Distillation loss weight): {self.alpha}")
        self.print_to_log_file(f"temperature (Distillation temperature): {self.temperature}")
        self.print_to_log_file(f"feature_reduction_factor (Feature reduction factor): {self.feature_reduction_factor}")
        self.print_to_log_file(f"Teacher model fold: {self.teacher_fold}")
        if self.rotate_training_folds:
            self.print_to_log_file(f"Fold rotation enabled with frequency: {self.rotate_folds_frequency} epochs")
            
    def load_teacher_model(self):
        """Load teacher model"""
        if self.teacher_model_folder is None:
            raise ValueError("Teacher model folder not set")
            
        self.print_to_log_file(f"Loading teacher model, folder: {self.teacher_model_folder}, folds: {self.teacher_fold}")
        
        # Clear teacher model list
        self.teacher_models = []
        
        # Load a teacher model for each fold
        for fold_idx in self.teacher_fold:
            self.print_to_log_file(f"Loading fold {fold_idx} teacher model...")
            
            # Create teacher model Predictor
            teacher_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            
            # Load teacher model
            teacher_checkpoint_path = join(self.teacher_model_folder, f"fold_{fold_idx}", self.teacher_checkpoint_name)
            teacher_predictor.initialize_from_trained_model_folder(
                self.teacher_model_folder,
                use_folds=(fold_idx,),
                checkpoint_name=self.teacher_checkpoint_name
            )
            
            # Get teacher model and set to eval mode
            teacher_model = teacher_predictor.network
            teacher_model.eval()
            
            # Ensure teacher model uses float32 precision and transfers to the correct device
            teacher_model = teacher_model.float().to(self.device)
            
            # Freeze teacher model parameters
            for param in teacher_model.parameters():
                param.requires_grad = False
                
            # Add to teacher model list
            self.teacher_models.append(teacher_model)
            
        self.print_to_log_file(f"Loaded {len(self.teacher_models)} teacher models")

    def build_network_architecture(self, 
                                 architecture_class_name: str = None,  # Ignore
                                 arch_init_kwargs: dict = None,  # Ignore 
                                 arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]] = None,  # Ignore
                                 num_input_channels: int = None,  # Can be used, but can also be obtained from plans
                                 num_output_channels: int = None,  # Can be used, but can also be obtained from plans
                                 enable_deep_supervision: bool = True):  # Default enable deep supervision
        """Rewrite network architecture building method, create lightweight student model, compatible with parent class parameter list"""
        
        # Determine student model type based on student_plans_identifier
        is_resenc_student = 'ResEnc' in getattr(self, 'student_plans_identifier', 'nnUNetPlans')
        
        if is_resenc_student:
            self.print_to_log_file("Building lightweight ResEnc student model...")
        else:
            self.print_to_log_file("Building lightweight standard UNet student model...")
        
        # If input/output channel numbers are not provided, obtain them from plans
        if num_input_channels is None:
            num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
        if num_output_channels is None:
            num_output_channels = self.label_manager.num_segmentation_heads
        
        # Check if there is a new format architecture field, if not, derive from configuration
        if 'architecture' in self.configuration_manager.configuration and 'arch_kwargs' in self.configuration_manager.configuration['architecture']:
            # New format plans
            plan_arch = self.configuration_manager.configuration['architecture']["arch_kwargs"]
            self.print_to_log_file("Using new version plans format network_architecture")
        else:
            # Old format plans, need to manually build arch_kwargs
            self.print_to_log_file("Detected old version plans format, manually build network parameters")
            
            # Get necessary parameters from configuration
            dim = len(self.configuration_manager.patch_size)
            n_stages = len(self.configuration_manager.pool_op_kernel_sizes) + 1
            
            # Base feature number and per stage feature number
            unet_max_num_features = self.plans_manager.plans.get('unet_max_num_features', 320)
            base_num_features = self.configuration_manager.configuration.get('UNet_base_num_features', 32)
            
            # Calculate feature number for each stage
            features_per_stage = [min(base_num_features * 2 ** i, unet_max_num_features) 
                                 for i in range(n_stages)]
            
            # Get other network parameters
            conv_kernel_sizes = self.configuration_manager.configuration.get(
                'conv_kernel_sizes', [[3,3,3]] * n_stages)
            
            # Build pool kernel size list, need to add a starting (1,1,1)
            pool_op_kernel_sizes = [(1,)*dim]
            for p in self.configuration_manager.pool_op_kernel_sizes:
                pool_op_kernel_sizes.append(p)
                
            # Get convolution layer number for each stage
            n_conv_per_stage = self.configuration_manager.configuration.get(
                'n_conv_per_stage_encoder', [2] * n_stages)
            n_conv_per_stage_decoder = self.configuration_manager.configuration.get(
                'n_conv_per_stage_decoder', [2] * (n_stages - 1))
            
            # Build plan_arch dictionary
            plan_arch = {
                "n_stages": n_stages,
                "features_per_stage": features_per_stage,
                "kernel_sizes": conv_kernel_sizes,
                "strides": pool_op_kernel_sizes,
                "n_conv_per_stage": n_conv_per_stage,
                "n_conv_per_stage_decoder": n_conv_per_stage_decoder,
                "conv_bias": True,
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "nonlin_kwargs": {"inplace": True}
            }
        
        # Reduce feature number
        lite_features_per_stage = [max(f // self.feature_reduction_factor, 8) for f in plan_arch["features_per_stage"]]
        self.print_to_log_file(f"Original feature number: {plan_arch['features_per_stage']}")
        self.print_to_log_file(f"Reduced feature number: {lite_features_per_stage}")
        
        # Create lightweight student model based on architecture type
        if is_resenc_student:
            # For ResEnc student, apply different block reduction strategies
            n_blocks_per_stage = plan_arch.get("n_blocks_per_stage", [1, 3, 4, 6, 6, 6][:plan_arch["n_stages"]])
            
            # Apply different strategies based on block_reduction_strategy parameter
            if self.block_reduction_strategy == 'reduce':
                # Strategy A: Reduce blocks by half (original approach)
                lite_n_blocks_per_stage = [max(n // 2, 1) for n in n_blocks_per_stage]
                strategy_desc = "reduced by half"
            elif self.block_reduction_strategy == 'keep':
                # Strategy B: Keep original blocks
                lite_n_blocks_per_stage = n_blocks_per_stage.copy()
                strategy_desc = "kept original"
            elif self.block_reduction_strategy == 'increase':
                # Strategy B+: Increase blocks by 1 per stage
                lite_n_blocks_per_stage = [min(n + 1, 8) for n in n_blocks_per_stage]
                strategy_desc = "increased by 1 per stage"
            elif self.block_reduction_strategy == 'adaptive':
                # Strategy B++: Adaptive increase based on compression ratio
                compression_ratios = [original/reduced for original, reduced in zip(plan_arch['features_per_stage'], lite_features_per_stage)]
                lite_n_blocks_per_stage = [min(n + max(0, int(ratio/4)), 8) for n, ratio in zip(n_blocks_per_stage, compression_ratios)]
                strategy_desc = "adaptively increased based on compression ratio"
            else:
                # Default: keep original
                lite_n_blocks_per_stage = n_blocks_per_stage.copy()
                strategy_desc = "kept original (default)"
            
            self.print_to_log_file(f"Original blocks per stage: {n_blocks_per_stage}")
            self.print_to_log_file(f"Student blocks per stage: {lite_n_blocks_per_stage} ({strategy_desc})")
            self.print_to_log_file(f"Block reduction strategy: {self.block_reduction_strategy}")
            
            network = LiteResEncStudent(
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                n_stages=plan_arch["n_stages"],
                features_per_stage=lite_features_per_stage,
                conv_op=Conv3d,
                kernel_sizes=[tuple(ks) if not isinstance(ks[0], (list, tuple)) else tuple(ks[0]) for ks in plan_arch["kernel_sizes"]],
                strides=[tuple(st) for st in plan_arch["strides"]],
                n_blocks_per_stage=lite_n_blocks_per_stage,
                n_conv_per_stage_decoder=plan_arch["n_conv_per_stage_decoder"],
                conv_bias=plan_arch["conv_bias"],
                norm_op=InstanceNorm3d,
                norm_op_kwargs=plan_arch["norm_op_kwargs"],
                nonlin=LeakyReLU,
                nonlin_kwargs=plan_arch["nonlin_kwargs"],
                deep_supervision=enable_deep_supervision
            )
        else:
            # Standard UNet student model
            network = LiteNNUNetStudent(
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                n_stages=plan_arch["n_stages"],
                features_per_stage=lite_features_per_stage,
                conv_op=Conv3d,
                kernel_sizes=[tuple(ks) if not isinstance(ks[0], (list, tuple)) else tuple(ks[0]) for ks in plan_arch["kernel_sizes"]],
                strides=[tuple(st) for st in plan_arch["strides"]],
                n_conv_per_stage=plan_arch["n_conv_per_stage"],
                n_conv_per_stage_decoder=plan_arch["n_conv_per_stage_decoder"],
                conv_bias=plan_arch["conv_bias"],
                norm_op=InstanceNorm3d,
                norm_op_kwargs=plan_arch["norm_op_kwargs"],
                nonlin=LeakyReLU,
                nonlin_kwargs=plan_arch["nonlin_kwargs"],
                deep_supervision=enable_deep_supervision
            )
        
        # Whether to compile
        if self._do_i_compile():
            self.print_to_log_file('Compiling network...')
            # Temporarily disable torch.compile, because it is not compatible with our distillation process
            # network = torch.compile(network)
            self.print_to_log_file('Due to compatibility issues, skip compiling network')
            
        return network
        
    def train_step(self, batch: dict) -> dict:
        """Distillation training step"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        if self.device.type == 'cuda':
            amp_context = autocast(device_type='cuda', enabled=True)
        else:
            amp_context = dummy_context()
            
        with amp_context:
            # Get all teacher model predictions and average (do not calculate gradients)
            all_teacher_logits = []
            with torch.no_grad():
                # Ensure data type matches, disable autocast
                with autocast(device_type='cuda', enabled=False):
                    for teacher_model in self.teacher_models:
                        teacher_output = teacher_model(data.float())
                        # If teacher uses deep supervision, only take the output of the highest resolution
                        if isinstance(teacher_output, (list, tuple)):
                            teacher_output = teacher_output[0]
                        all_teacher_logits.append(teacher_output)
                
                # Calculate average of teacher model predictions
                if len(all_teacher_logits) > 1:
                    teacher_logits = torch.mean(torch.stack(all_teacher_logits), dim=0)
                else:
                    teacher_logits = all_teacher_logits[0]
                
            # Get student model prediction (need to calculate gradients)
            with autocast(device_type='cuda', enabled=True):
                student_logits = self.network(data.float())
            
            # Calculate segmentation loss
            # If deep supervision is enabled, need to wrap student_logits as list
            if self.enable_deep_supervision:
                if not isinstance(student_logits, (list, tuple)):
                    student_logits_wrapped = [student_logits]
                else:
                    student_logits_wrapped = student_logits
                
                if not isinstance(target, (list, tuple)):
                    target_wrapped = [target]
                else:
                    target_wrapped = target
                    
                # Calculate segmentation loss
                seg_loss = self.loss(student_logits_wrapped, target_wrapped)
            else:
                # Do not use deep supervision, directly calculate loss
                seg_loss = self.loss(student_logits, target)
            
            # Calculate distillation loss (KL divergence)
            # For deep supervision case, use the output of the highest resolution
            if isinstance(student_logits, (list, tuple)):
                student_logits_for_distill = student_logits[0]
            else:
                student_logits_for_distill = student_logits
                
            # Use predefined distillation loss function to calculate loss
            distill_loss = distillation_loss_fn(student_logits_for_distill, teacher_logits, self.temperature)
            
            # Total loss = Segmentation loss * (1-alpha) + Distillation loss * alpha
            total_loss = seg_loss * (1 - self.alpha) + distill_loss * self.alpha
            
            # Use gradient scaler for backpropagation
            if self.grad_scaler is not None:
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
            
            # Record loss
            if self.logger is not None:
                # If list does not exist, create it first
                if 'train_seg_losses' not in self.logger.my_fantastic_logging:
                    self.logger.my_fantastic_logging['train_seg_losses'] = []
                if 'train_distill_losses' not in self.logger.my_fantastic_logging:
                    self.logger.my_fantastic_logging['train_distill_losses'] = []
                
                self.logger.my_fantastic_logging['train_seg_losses'].append(seg_loss.detach().cpu().numpy())
                self.logger.my_fantastic_logging['train_distill_losses'].append(distill_loss.detach().cpu().numpy())
            
            # Return parent class expected format
            return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """Validation step, compatible with student model without deep supervision output"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if self.device.type == 'cuda':
            amp_context = autocast(device_type='cuda', enabled=True)
        else:
            amp_context = dummy_context()
            
        with amp_context:
            # Get student model prediction
            with autocast(device_type='cuda', enabled=False):
                output = self.network(data.float())
            del data
            
            # If deep supervision is enabled but model output is not list
            if self.enable_deep_supervision:
                if not isinstance(output, (list, tuple)):
                    # Wrap output as list to be compatible with deep supervision loss
                    output_wrapped = [output]
                else:
                    output_wrapped = output
                
                # Ensure target is also list
                if not isinstance(target, (list, tuple)):
                    target_wrapped = [target]
                else:
                    target_wrapped = target
                    
                # Calculate segmentation loss
                l = self.loss(output_wrapped, target_wrapped)
                
                # For subsequent evaluation, get the output of the highest resolution
                output = output_wrapped[0]
                target = target_wrapped[0]
            else:
                # Do not use deep supervision, directly calculate loss
                l = self.loss(output, target)

        # From here on, it's parent class code copy, for generating evaluation metrics
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """Training epoch end processing, only use loss"""
        outputs = collate_outputs(train_outputs)
        
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        
        
    def on_epoch_end(self):
        """epoch end processing, increase distillation loss record and handle fold rotation"""
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        
        # Calculate and print average segmentation and distillation loss
        if 'train_seg_losses' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['train_seg_losses']) > 0:
            avg_seg_loss = np.mean(self.logger.my_fantastic_logging['train_seg_losses'])
            self.print_to_log_file('avg_seg_loss', np.round(avg_seg_loss, decimals=4))
            # Clear list, prepare for next epoch
            self.logger.my_fantastic_logging['train_seg_losses'] = []
            
        if 'train_distill_losses' in self.logger.my_fantastic_logging and len(self.logger.my_fantastic_logging['train_distill_losses']) > 0:
            avg_distill_loss = np.mean(self.logger.my_fantastic_logging['train_distill_losses'])
            self.print_to_log_file('avg_distill_loss', np.round(avg_distill_loss, decimals=4))
            # Clear list, prepare for next epoch
            self.logger.my_fantastic_logging['train_distill_losses'] = []
            
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
            
        # Handle fold rotation if enabled
        fold_updated = self.update_fold_for_next_rotation()
        if fold_updated:
            self.print_to_log_file(f"Fold rotated, now training with fold {self.fold}")

        self.current_epoch += 1
        
    def load_student_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        Overridden checkpoint loading method specifically for handling student model loading
        """
        # Check if already initialized
        if not self.was_initialized:
            raise RuntimeError("Must call initialize() before loading checkpoint! Please make sure to call the initialize() method first.")
            
        self.print_to_log_file(f"Loading checkpoint from {filename_or_checkpoint if isinstance(filename_or_checkpoint, str) else 'dict'}")
        
        # Load checkpoint
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        # Restore training state from checkpoint
        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        if 'inference_allowed_mirroring_axes' in checkpoint.keys():
            self.inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
            
        # Get network weights
        new_state_dict = {}
        
        # Determine if we need to handle OptimizedModule
        is_optimized = isinstance(self.network, OptimizedModule) if not self.is_ddp else isinstance(self.network.module, OptimizedModule)
        
        for k, value in checkpoint['network_weights'].items():
            key = k
            # Handle weights created by DataParallel
            if key.startswith('module.'):
                key = key[7:]
            # Handle weights created by torch.compile (OptimizedModule)
            # Remove _orig_mod prefix from checkpoint if present, we'll load to _orig_mod directly
            if key.startswith('_orig_mod.'):
                key = key[10:]  # Remove '_orig_mod.' prefix
            # Add to new state dictionary
            new_state_dict[key] = value
        
        # Check if model structure matches
        # For OptimizedModule, we need to compare against _orig_mod's state_dict
        if is_optimized:
            if self.is_ddp:
                actual_model_keys = set(self.network.module._orig_mod.state_dict().keys())
            else:
                actual_model_keys = set(self.network._orig_mod.state_dict().keys())
        else:
            if self.is_ddp:
                actual_model_keys = set(self.network.module.state_dict().keys())
            else:
                actual_model_keys = set(self.network.state_dict().keys())
        
        checkpoint_keys = set(new_state_dict.keys())
        
        # Check missing keys
        missing_keys = actual_model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - actual_model_keys
        
        # Calculate compatibility statistics
        compatible_keys = actual_model_keys & checkpoint_keys
        compatible_params = {k: v for k, v in new_state_dict.items() if k in actual_model_keys}
        
        total_model_params = len(actual_model_keys)
        total_compatible_params = len(compatible_params)
        compatibility_ratio = total_compatible_params / total_model_params if total_model_params > 0 else 0
        
        self.print_to_log_file(f"Model compatibility analysis:")
        self.print_to_log_file(f"  Total model parameters: {total_model_params}")
        self.print_to_log_file(f"  Compatible parameters: {total_compatible_params}")
        self.print_to_log_file(f"  Compatibility ratio: {compatibility_ratio:.2%}")
        
        if len(missing_keys) > 0:
            self.print_to_log_file(f"  Parameters missing in checkpoint: {len(missing_keys)}")
            # Only show first 10 missing keys to avoid cluttering the log
            if len(missing_keys) <= 10:
                self.print_to_log_file(f"  Missing keys: {missing_keys}")
            else:
                missing_sample = list(missing_keys)[:10]
                self.print_to_log_file(f"  Missing keys (first 10): {missing_sample}")
                self.print_to_log_file(f"  ... and {len(missing_keys) - 10} more")
                
        if len(unexpected_keys) > 0:
            self.print_to_log_file(f"  Unexpected keys in checkpoint: {len(unexpected_keys)}")
            # Only show first 10 unexpected keys to avoid cluttering the log
            if len(unexpected_keys) <= 10:
                self.print_to_log_file(f"  Unexpected keys: {unexpected_keys}")
            else:
                unexpected_sample = list(unexpected_keys)[:10]
                self.print_to_log_file(f"  Unexpected keys (first 10): {unexpected_sample}")
                self.print_to_log_file(f"  ... and {len(unexpected_keys) - 10} more")
        
        # Always use partial loading for distillation models to handle architecture differences
        self.print_to_log_file("Loading compatible parameters (partial loading for architecture flexibility)...")
        
        # Debug: check current network keys
        if is_optimized:
            if self.is_ddp:
                debug_keys = list(self.network.module._orig_mod.state_dict().keys())[:3]
            else:
                debug_keys = list(self.network._orig_mod.state_dict().keys())[:3]
        else:
            if self.is_ddp:
                debug_keys = list(self.network.module.state_dict().keys())[:3]
            else:
                debug_keys = list(self.network.state_dict().keys())[:3]
        self.print_to_log_file(f"Target network keys (first 3): {debug_keys}")
        self.print_to_log_file(f"Checkpoint keys (first 3): {list(new_state_dict.keys())[:3]}")
        self.print_to_log_file(f"Network is OptimizedModule: {is_optimized}")
        
        # Load the state dict directly with strict=False to allow partial loading
        try:
            if self.is_ddp:
                if isinstance(self.network.module, OptimizedModule):
                    result = self.network.module._orig_mod.load_state_dict(new_state_dict, strict=False)
                else:
                    result = self.network.module.load_state_dict(new_state_dict, strict=False)
            else:
                if isinstance(self.network, OptimizedModule):
                    result = self.network._orig_mod.load_state_dict(new_state_dict, strict=False)
                else:
                    result = self.network.load_state_dict(new_state_dict, strict=False)
            
            self.print_to_log_file(f"Load state dict result - Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
                    
            self.print_to_log_file(f"Successfully loaded {total_compatible_params} compatible parameters out of {total_model_params} total parameters")
            
            if compatibility_ratio < 0.5:
                self.print_to_log_file(f"Warning: Low compatibility ratio ({compatibility_ratio:.2%}). Model architecture may be significantly different.")
                self.print_to_log_file("Consider starting training from scratch if performance is poor.")
            elif compatibility_ratio < 0.8:
                self.print_to_log_file(f"Notice: Moderate compatibility ratio ({compatibility_ratio:.2%}). Some architecture differences detected.")
            else:
                self.print_to_log_file(f"Good compatibility ratio ({compatibility_ratio:.2%}). Most parameters loaded successfully.")
                
        except Exception as e:
            self.print_to_log_file(f"Error during partial loading: {e}")
            self.print_to_log_file("Will initialize with random weights for incompatible parts")
            raise e
            
        # Load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.print_to_log_file("Successfully loaded optimizer state")
        except Exception as e:
            self.print_to_log_file(f"Error loading optimizer state, will use new optimizer: {e}")
        
        # Load gradient scaler state
        if self.grad_scaler is not None and 'grad_scaler_state' in checkpoint and checkpoint['grad_scaler_state'] is not None:
            try:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
                self.print_to_log_file("Successfully loaded gradient scaler state")
            except Exception as e:
                self.print_to_log_file(f"Error loading gradient scaler state: {e}")
                
        self.print_to_log_file(f"Resuming training from epoch {self.current_epoch}")


class nnUNetDistillationTrainerDA5(nnUNetDistillationTrainer, nnUNetTrainerDA5):
    """
    Knowledge Distillation Trainer with DA5 Strong Data Augmentation
    
    Simple multiple inheritance approach - inherits distillation from nnUNetDistillationTrainer
    and DA5 data augmentation from nnUNetTrainerDA5.
    """
    
    def __init__(self, plans, configuration, fold, dataset_json, 
                 teacher_model_folder=None, 
                 teacher_fold=0,
                 teacher_checkpoint_name='checkpoint_final.pth',
                 alpha=0.3,
                 temperature=3.0,
                 feature_reduction_factor=2,
                 block_reduction_strategy='keep',
                 rotate_training_folds=False,
                 rotate_folds_frequency=5,
                 device=torch.device('cuda')):
        """
        Initialize the DA5-enhanced distillation trainer
        
        Parameters are the same as nnUNetDistillationTrainer, but this trainer
        uses the strong data augmentation from DA5 for better performance on small datasets.
        """
        # Initialize the base distillation trainer (this handles all distillation logic)
        nnUNetDistillationTrainer.__init__(
            self, plans, configuration, fold, dataset_json,
            teacher_model_folder, teacher_fold, teacher_checkpoint_name,
            alpha, temperature, feature_reduction_factor, block_reduction_strategy,
            rotate_training_folds, rotate_folds_frequency, device
        )
        
        self.print_to_log_file("Using DA5 strong data augmentation for knowledge distillation") 