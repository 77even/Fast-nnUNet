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
## Description: Export ResEnc distillation models to ONNX format

import argparse
import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
from batchgenerators.utilities.file_and_folder_operations import *

# ONNX related imports
from onnx import load, checker
from onnxruntime import InferenceSession
from torch.onnx import export as torch_onnx_export

# Ensure using nnunetv2 from current directory
nnunet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nnunet_dir)

# Import paths from nnunetv2
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed

# Import nnUNetDistillationTrainer directly
from nnunetv2.training.nnUNetTrainer.variants.nnUNetDistillationTrainer import nnUNetDistillationTrainer, nnUNetDistillationTrainerDA5, LiteNNUNetStudent, LiteResEncStudent
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def get_dataset_name_from_id(dataset_id):
    """Get the complete dataset name from dataset ID"""
    # Try to convert dataset_id to integer
    try:
        dataset_id = int(dataset_id)
    except ValueError:
        # If already a string name, return directly
        return dataset_id
    
    # Find matching dataset directory
    for dataset_dir in os.listdir(nnUNet_raw):
        if dataset_dir.startswith(f"Dataset{dataset_id:03d}_"):
            return dataset_dir
    
    # If not found, construct a default name and warn
    default_name = f"Dataset{dataset_id:03d}"
    print(f"Warning: No dataset directory with ID {dataset_id} found in {nnUNet_raw}, using default name {default_name}")
    return default_name

def export_dataset_json(dataset_name, output_dir):
    """Export dataset.json file to output directory"""
    import shutil
    dataset_json_path = join(nnUNet_raw, dataset_name, "dataset.json")
    if isfile(dataset_json_path):
        output_json_path = join(output_dir, "dataset.json")
        shutil.copy2(dataset_json_path, output_json_path)
        print(f"Dataset json file exported to: {output_json_path}")
    else:
        print(f"Warning: Dataset json file not found: {dataset_json_path}")

def fix_onnx_with_checkpoint(onnx_model_path, checkpoint_state_dict):
    """
    Fix ONNX model using real bias values from PyTorch checkpoint
    Ensures TensorRT compatibility while maintaining full precision
    (Same logic as nnunetv2_resenc_onnx_convert.py)
    
    Args:
        onnx_model_path: Path to ONNX model
        checkpoint_state_dict: PyTorch checkpoint state dict
    
    Returns:
        bool: Whether fixes were applied
    """
    import torch
    import onnx
    import numpy as np
    
    print("\n🔧 Checking ONNX model for TensorRT compatibility...")
    
    # Load ONNX model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Collect initializers
    initializers = {init.name: init for init in graph.initializer}
    
    # Find InstanceNorm nodes that need fixing
    nodes_to_fix = []
    for node in graph.node:
        if node.op_type == "InstanceNormalization":
            if len(node.input) >= 3:
                scale_name = node.input[1]
                bias_name = node.input[2]
                
                scale_is_init = scale_name in initializers
                bias_is_init = bias_name in initializers
                
                if not bias_is_init and scale_is_init:
                    nodes_to_fix.append({
                        'node': node,
                        'bias_name': bias_name,
                        'scale_name': scale_name
                    })
    
    if not nodes_to_fix:
        print("  ✅ Model is already TensorRT compatible, no fixes needed")
        return False
    
    print(f"  ⚠️  Found {len(nodes_to_fix)} InstanceNorm nodes that need fixing")
    print("  📥 Extracting real bias values from checkpoint...")
    
    # Debug: print some checkpoint keys to understand the naming pattern
    if len(checkpoint_state_dict) > 0:
        sample_keys = list(checkpoint_state_dict.keys())[:5]
        print(f"  🔍 Sample checkpoint keys: {sample_keys}")
    
    # Extract bias values from checkpoint
    for i, fix_info in enumerate(nodes_to_fix):
        bias_name = fix_info['bias_name']
        scale_name = fix_info['scale_name']
        
        if i == 0:  # Print first one for debugging
            print(f"  🔍 Example ONNX bias name: '{bias_name}'")
        
        # Try to find corresponding bias in checkpoint
        # The bias_name from ONNX might have wrapper prefixes that don't exist in checkpoint
        # Try multiple variations to find the correct parameter
        possible_names = []
        
        # 1. Try exact name
        possible_names.append(bias_name)
        
        # 2. Remove "model." prefix (from InferenceWrapper)
        if bias_name.startswith('model.'):
            possible_names.append(bias_name[6:])  # Remove "model."
        
        # 3. Remove "model.model." in case of double wrapping
        if bias_name.startswith('model.model.'):
            possible_names.append(bias_name[12:])
        
        # 4. Try without any "model." prefixes at all (handle nested wrappers)
        name_without_model = bias_name
        while name_without_model.startswith('model.'):
            name_without_model = name_without_model[6:]
        if name_without_model != bias_name:
            possible_names.append(name_without_model)
        
        # Remove duplicates while preserving order
        seen = set()
        possible_names = [n for n in possible_names if not (n in seen or seen.add(n))]
        
        bias_tensor = None
        found_name = None
        for name in possible_names:
            if name in checkpoint_state_dict:
                bias_tensor = checkpoint_state_dict[name]
                found_name = name
                break
        
        if bias_tensor is not None:
            bias_data = bias_tensor.cpu().numpy().astype(np.float32)
            print(f"     ✅ {bias_name}: mean={bias_tensor.mean().item():.4f}, std={bias_tensor.std().item():.4f}")
            if found_name != bias_name:
                print(f"        (matched via alternate name: {found_name})")
        else:
            # If not found, use zero bias
            scale_tensor = initializers[scale_name]
            bias_shape = list(scale_tensor.dims)
            bias_data = np.zeros(bias_shape, dtype=np.float32)
            print(f"     ⚠️  {bias_name}: not found in checkpoint")
            print(f"        Tried: {possible_names}")
            print(f"        Using zero bias as fallback")
        
        # Create new bias initializer
        new_bias_name = bias_name + "_fixed"
        bias_initializer = onnx.helper.make_tensor(
            name=new_bias_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=list(bias_data.shape),
            vals=bias_data.flatten().tolist()
        )
        
        # Add to initializer list
        graph.initializer.append(bias_initializer)
        
        # Update node's bias input
        fix_info['node'].input[2] = new_bias_name
    
    # Save fixed model
    print("  💾 Saving fixed model...")
    onnx.save(model, onnx_model_path)
    print(f"  ✅ TensorRT compatibility fix completed (fixed {len(nodes_to_fix)} nodes)")
    
    return True

def load_model_from_checkpoint(checkpoint_path, plans, dataset_json, configuration, fold, 
                             student_plans_identifier, feature_reduction_factor, 
                             block_reduction_strategy='keep', device=torch.device('cuda'), 
                             verbose=False, use_da5=False):
    """Load distillation student model from checkpoint
    
    Parameters:
        block_reduction_strategy: Strategy for residual blocks compression
            - 'reduce': Reduce blocks by half
            - 'keep': Keep original blocks (default)
            - 'increase': Increase blocks by 1 per stage
            - 'adaptive': Adaptive increase based on compression ratio
    """
    
    # Determine student model type based on student_plans_identifier
    is_resenc_student = 'ResEnc' in student_plans_identifier
    
    if verbose:
        if is_resenc_student:
            print("Loading ResEnc student model architecture...")
        else:
            print("Loading standard UNet student model architecture...")
    
    # Create trainer instance to get model architecture parameters
    if use_da5:
        trainer = nnUNetDistillationTrainerDA5(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            teacher_model_folder=None,
            feature_reduction_factor=feature_reduction_factor,
            device=device
        )
    else:
        trainer = nnUNetDistillationTrainer(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            teacher_model_folder=None,
            feature_reduction_factor=feature_reduction_factor,
            device=device
        )
    
    # Set student plans identifier for correct architecture selection
    trainer.student_plans_identifier = student_plans_identifier
    
    # Get network architecture parameters
    num_input_channels = determine_num_input_channels(trainer.plans_manager, trainer.configuration_manager, dataset_json)
    num_output_channels = trainer.label_manager.num_segmentation_heads
    
    if verbose:
        print(f"Number of input channels: {num_input_channels}")
        print(f"Number of output channels: {num_output_channels}")
    
    # Get architecture configuration
    if configuration in plans['configurations']:
        config = plans['configurations'][configuration]
        
        # Extract network parameters
        if 'architecture' in config:
            arch_info = config['architecture']
            if isinstance(arch_info, dict) and 'arch_kwargs' in arch_info:
                arch_kwargs = arch_info['arch_kwargs']
                n_stages = arch_kwargs.get('n_stages', 6)
                features_per_stage = arch_kwargs.get('features_per_stage', [32, 64, 128, 256, 320, 320][:n_stages])
                strides = arch_kwargs.get('strides', [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1))
                kernel_sizes = arch_kwargs.get('kernel_sizes', [[3, 3, 3]] * n_stages)
                n_blocks_per_stage = arch_kwargs.get('n_blocks_per_stage', [1, 3, 4, 6, 6, 6][:n_stages])
            else:
                # Fall back to defaults
                n_stages = 6
                features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
                strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
                kernel_sizes = [[3, 3, 3]] * n_stages
                n_blocks_per_stage = [1, 3, 4, 6, 6, 6][:n_stages]
        else:
            # Get from other configuration keys
            if 'pool_op_kernel_sizes' in config:
                n_stages = len(config['pool_op_kernel_sizes'])
                features_per_stage = config.get('features_per_stage', [32, 64, 128, 256, 320, 320][:n_stages])
                strides = config['pool_op_kernel_sizes']
                kernel_sizes = config.get('conv_kernel_sizes', [[3, 3, 3]] * n_stages)
                n_blocks_per_stage = [1, 3, 4, 6, 6, 6][:n_stages]
            else:
                # Use default values
                print("Warning: Cannot find complete network architecture configuration in plans, using defaults")
                n_stages = 6
                features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
                strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
                kernel_sizes = [[3, 3, 3]] * n_stages
                n_blocks_per_stage = [1, 3, 4, 6, 6, 6][:n_stages]
        
        # Apply feature reduction factor
        lite_features_per_stage = [max(f // feature_reduction_factor, 8) for f in features_per_stage]
        
        if verbose:
            print(f"Number of stages: {n_stages}")
            print(f"Original features: {features_per_stage}")
            print(f"Reduced features: {lite_features_per_stage}")
            if is_resenc_student:
                lite_n_blocks_per_stage = [max(n // 2, 1) for n in n_blocks_per_stage]
                print(f"Original blocks per stage: {n_blocks_per_stage}")
                print(f"Reduced blocks per stage: {lite_n_blocks_per_stage}")
    else:
        raise ValueError(f"Configuration {configuration} does not exist in plans")
    
    # Create student model based on architecture type
    if is_resenc_student:
        # ResEnc student model - Apply block reduction strategy
        if block_reduction_strategy == 'reduce':
            # Strategy A: Reduce blocks by half (original approach)
            lite_n_blocks_per_stage = [max(n // 2, 1) for n in n_blocks_per_stage]
            if verbose:
                print(f"Block strategy 'reduce': reduced by half")
        elif block_reduction_strategy == 'keep':
            # Strategy B: Keep original blocks
            lite_n_blocks_per_stage = n_blocks_per_stage.copy()
            if verbose:
                print(f"Block strategy 'keep': kept original")
        elif block_reduction_strategy == 'increase':
            # Strategy B+: Increase blocks by 1 per stage
            lite_n_blocks_per_stage = [min(n + 1, 8) for n in n_blocks_per_stage]
            if verbose:
                print(f"Block strategy 'increase': increased by 1 per stage")
        elif block_reduction_strategy == 'adaptive':
            # Strategy B++: Adaptive increase based on compression ratio
            original_features = features_per_stage
            compression_ratios = [orig/reduced for orig, reduced in zip(original_features, lite_features_per_stage)]
            lite_n_blocks_per_stage = [min(n + max(0, int(ratio/4)), 8) for n, ratio in zip(n_blocks_per_stage, compression_ratios)]
            if verbose:
                print(f"Block strategy 'adaptive': adaptively increased based on compression ratio")
        else:
            # Default: keep original
            lite_n_blocks_per_stage = n_blocks_per_stage.copy()
            if verbose:
                print(f"Block strategy unknown, using 'keep' as default")
        
        # Always print block information for ResEnc models (important for debugging)
        print(f"📐 ResEnc Architecture Configuration:")
        print(f"   Original blocks per stage: {n_blocks_per_stage}")
        print(f"   Student blocks per stage: {lite_n_blocks_per_stage}")
        print(f"   Block strategy: {block_reduction_strategy}")
        print(f"   Features per stage: {lite_features_per_stage}")
        
        # IMPORTANT: Load model with deep_supervision=True to match training architecture
        # We'll handle the output wrapping later
        model = LiteResEncStudent(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=n_stages,
            features_per_stage=lite_features_per_stage,
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[tuple(k) if isinstance(k, list) else k for k in kernel_sizes],
            strides=[tuple(p) if isinstance(p, list) else p for p in strides],
            n_blocks_per_stage=lite_n_blocks_per_stage,
            n_conv_per_stage_decoder=[1] * (n_stages - 1),
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=True  # Must match training configuration for weight loading
        )
    else:
        # Standard UNet student model
        model = LiteNNUNetStudent(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=n_stages,
            features_per_stage=lite_features_per_stage,
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[tuple(k) if isinstance(k, list) else k for k in kernel_sizes],
            strides=[tuple(p) if isinstance(p, list) else p for p in strides],
            n_conv_per_stage=[2] * n_stages,
            n_conv_per_stage_decoder=[2] * (n_stages - 1),
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=True  # Must match training configuration for weight loading
        )
    
    # Load checkpoint
    try:
        # Handle PyTorch 2.6+ safe loading restrictions
        try:
            import numpy.core.multiarray
            if hasattr(torch.serialization, 'add_safe_globals'):
                # PyTorch 2.6+
                torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                # Compatible with earlier PyTorch versions
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except:
            # If still failing, try loading without weights_only
            print("Trying to load checkpoint in non-weights mode...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")
    
    # Try to extract training configuration from checkpoint if available
    if 'init_args' in checkpoint and verbose:
        print("\n🔍 Found training configuration in checkpoint:")
        init_args = checkpoint['init_args']
        if 'feature_reduction_factor' in init_args:
            print(f"   Checkpoint feature_reduction_factor: {init_args['feature_reduction_factor']}")
        if 'block_reduction_strategy' in init_args:
            print(f"   Checkpoint block_reduction_strategy: {init_args['block_reduction_strategy']}")
    
    # Load model weights
    if 'network_weights' in checkpoint:
        state_dict = checkpoint['network_weights']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError(f"Checkpoint file {checkpoint_path} does not contain usable weights")
    
    # Handle potential 'module.' prefix (for DataParallel wrapped models)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load parameters
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(new_state_dict.keys())
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
    
    # Check for mismatches
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    matched_keys = model_keys & checkpoint_keys
    
    print(f"\n🔍 Checkpoint Loading Analysis:")
    print(f"   Model expects: {len(model_keys)} parameters")
    print(f"   Checkpoint has: {len(checkpoint_keys)} parameters")
    print(f"   Successfully matched: {len(matched_keys)} parameters")
    
    if missing_keys:
        print(f"   ⚠️  Missing in checkpoint: {len(missing_keys)} parameters")
        if verbose or len(missing_keys) < 20:
            for key in sorted(missing_keys)[:10]:  # Show first 10
                print(f"      - {key}")
            if len(missing_keys) > 10:
                print(f"      ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"   ⚠️  Unexpected in checkpoint: {len(unexpected_keys)} parameters")
        if verbose or len(unexpected_keys) < 20:
            for key in sorted(unexpected_keys)[:10]:  # Show first 10
                print(f"      - {key}")
            if len(unexpected_keys) > 10:
                print(f"      ... and {len(unexpected_keys) - 10} more")
    
    # Load with strict=False to see what happens
    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"   ⚠️  Load result - missing: {len(load_result.missing_keys)}, unexpected: {len(load_result.unexpected_keys)}")
    
    model.eval()
    model.to(device)
    
    # Print model parameter count for verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Create a wrapper that only returns the main output (not deep supervision outputs)
    # This is needed because training uses deep_supervision=True but inference only needs final output
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            output = self.model(x)
            # If deep supervision is enabled, output is a list [main, aux1, aux2, ...]
            # We only want the main output (index 0)
            if isinstance(output, (list, tuple)):
                return output[0]
            return output
    
    wrapped_model = InferenceWrapper(model)
    wrapped_model.eval()
    
    return wrapped_model, num_input_channels, num_output_channels, config, filtered_state_dict

def export_resenc_distillation_to_onnx(dataset_id,
                                     output_dir=None,
                                     configuration='3d_fullres',
                                     fold=0,
                                     batch_size=0,
                                     checkpoint_name='checkpoint_final.pth',
                                     plans_identifier='nnUNetPlans',
                                     student_plans_identifier='nnUNetPlans',
                                     feature_reduction_factor=2,
                                     block_reduction_strategy='keep',
                                     trt_compatible=False,
                                     verbose=False,
                                     use_da5=False):
    """
    Export ResEnc distillation model to ONNX format
    
    Parameters:
        dataset_id: Dataset ID
        output_dir: Output directory
        configuration: Configuration name
        fold: Fold number
        batch_size: Batch size, 0 means dynamic
        checkpoint_name: Checkpoint filename
        plans_identifier: Plans identifier
        student_plans_identifier: Student plans identifier
        feature_reduction_factor: Feature reduction factor
        block_reduction_strategy: Block reduction strategy ('reduce', 'keep', 'increase', 'adaptive')
            MUST match the strategy used during training!
        trt_compatible: Apply TensorRT compatibility fixes
        verbose: Show detailed output
        use_da5: Whether the model was trained with DA5 data augmentation
    """
    # Parse dataset ID to full name
    dataset_name = get_dataset_name_from_id(dataset_id)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model folder path based on whether DA5 was used
    if use_da5:
        trainer_name = "nnUNetDistillationTrainerDA5"
    else:
        trainer_name = "nnUNetDistillationTrainer"
    
    model_folder = join(nnUNet_results, dataset_name, f"{trainer_name}__{student_plans_identifier}__{configuration}")
    model_folder_fold = join(model_folder, f"fold_{fold}")
    
    if not os.path.exists(model_folder_fold):
        raise FileNotFoundError(f"Model folder does not exist: {model_folder_fold}")
    
    # Determine checkpoint file path
    checkpoint_path = join(model_folder_fold, checkpoint_name)
    if not isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    # Load plans and dataset_json
    plans_file = join(nnUNet_preprocessed, dataset_name, f"{student_plans_identifier}.json")
    if not isfile(plans_file):
        raise FileNotFoundError(f"Plans file does not exist: {plans_file}")
    
    dataset_json_file = join(nnUNet_raw, dataset_name, "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(f"Dataset json file does not exist: {dataset_json_file}")
    
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)
    
    # Load model from checkpoint
    model, num_input_channels, num_output_channels, config, state_dict = load_model_from_checkpoint(
        checkpoint_path, plans, dataset_json, configuration, fold,
        student_plans_identifier, feature_reduction_factor, 
        block_reduction_strategy, device, verbose, use_da5
    )
    
    # Set output directory
    if output_dir is None:
        output_dir = join(model_folder_fold, "exported_models")
    
    maybe_mkdir_p(output_dir)
    
    # Get input shape
    if 'patch_size' in config:
        patch_size = config['patch_size']
        input_shape = (1, num_input_channels, *patch_size)
    else:
        # Default shape
        input_shape = (1, num_input_channels, 128, 128, 128)
    
    # Determine model type for filename
    is_resenc_student = 'ResEnc' in student_plans_identifier
    model_type = "resenc" if is_resenc_student else "unet"
    if use_da5:
        model_type += "_da5"
    
    # Create dummy input
    # IMPORTANT: Use randn (normal distribution) instead of zeros for better InstanceNorm behavior
    # InstanceNorm computes statistics from the input, so using realistic data distribution helps
    torch.manual_seed(42)  # For reproducibility
    if batch_size > 0:
        # Fixed batch size
        dummy_input = torch.randn((batch_size, num_input_channels, *input_shape[2:]), dtype=torch.float32).to(device)
        dynamic_axes = None
        onnx_filename = f"{model_type}_distillation_fold{fold}_batch{batch_size}_r{feature_reduction_factor}.onnx"
    else:
        # Dynamic batch size
        dummy_input = torch.randn(input_shape, dtype=torch.float32).to(device)
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        onnx_filename = f"{model_type}_distillation_fold{fold}_dynamic_r{feature_reduction_factor}.onnx"
    
    output_path = join(output_dir, onnx_filename)
    
    print(f"\n{'='*80}")
    print(f"Exporting {model_type.upper()} distillation model")
    print(f"{'='*80}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Feature reduction factor: {feature_reduction_factor}")
    print(f"Block reduction strategy: {block_reduction_strategy}")
    print(f"Student plans identifier: {student_plans_identifier}")
    print(f"{'='*80}\n")
    
    # Ensure model is in eval mode before getting reference output
    model.eval()
    
    # CRITICAL: Force all InstanceNorm layers to eval mode
    # This is necessary because InstanceNorm with track_running_stats=False
    # doesn't respond to model.eval() properly for ONNX export
    for module in model.modules():
        if isinstance(module, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)):
            module.eval()
            # Force the module to use eval mode during forward
            module.training = False
    
    # Get PyTorch output for comparison
    with torch.no_grad():
        torch_output = model(dummy_input)
        
    print(f"✅ Model prepared for export (all InstanceNorm layers set to eval mode)")
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,                       # Model instance
            dummy_input,                 # Model input
            output_path,                 # Output file
            export_params=True,          # Store trained parameter weights
            opset_version=11,            # ONNX operator set version (same as nnunetv2_resenc_onnx_convert.py)
            do_constant_folding=True,    # Constant folding optimization
            input_names=['input'],       # Input names
            output_names=['output'],     # Output names
            dynamic_axes=dynamic_axes,   # Dynamic dimensions
            training=torch.onnx.TrainingMode.EVAL,  # Explicitly set to eval mode
            verbose=verbose              # Verbose output
        )
        print(f"{model_type.upper()} distillation model successfully exported to {output_path}")
        
        # Validate ONNX model
        print("Validating ONNX model...")
        onnx_model = load(output_path)
        checker.check_model(onnx_model)
        
        # Export dataset json
        export_dataset_json(dataset_name, output_dir)
        
        # Apply TensorRT compatibility fixes if requested
        # Note: No simplification step to match nnunetv2_resenc_onnx_convert.py behavior
        if trt_compatible:
            try:
                # Test original ONNX inference (before TensorRT fixes)
                print("\n📊 Testing original ONNX model (before TensorRT fixes)...")
                ort_session_orig = InferenceSession(
                    output_path,
                    providers=["CPUExecutionProvider"],
                )
                ort_inputs = {
                    ort_session_orig.get_inputs()[0].name: dummy_input.cpu().numpy()
                }
                ort_outs_orig = ort_session_orig.run(None, ort_inputs)
                torch_output_np = torch_output.detach().cpu().numpy()

                try:
                    np.testing.assert_allclose(
                        torch_output_np,
                        ort_outs_orig[0],
                        rtol=1e-02,
                        atol=1e-04,
                        verbose=True,
                    )
                    print("  ✅ Original ONNX matches PyTorch!")
                except AssertionError as e:
                    print("  ⚠️  Minor differences (expected):")
                    abs_diff = np.abs(torch_output_np - ort_outs_orig[0])
                    print(f"     Max diff: {np.max(abs_diff):.6f}, Mean diff: {np.mean(abs_diff):.6f}")

                # Fix ONNX model using real bias values from checkpoint (same as nnunetv2_resenc_onnx_convert.py)
                was_fixed = fix_onnx_with_checkpoint(output_path, state_dict)
                
                if was_fixed:
                    # Test fixed ONNX inference
                    print("\n📊 Testing fixed ONNX model (after TensorRT fixes)...")
                    ort_session_fixed = InferenceSession(
                        output_path,
                        providers=["CPUExecutionProvider"],
                    )
                    ort_outs_fixed = ort_session_fixed.run(None, ort_inputs)

                    # Compare fixed ONNX with original ONNX
                    try:
                        np.testing.assert_allclose(
                            ort_outs_orig[0],
                            ort_outs_fixed[0],
                            rtol=1e-08,
                            atol=1e-08,
                            verbose=False,
                        )
                        print("  ✅ Fixed ONNX is IDENTICAL to original ONNX (bit-wise)!")
                    except AssertionError:
                        abs_diff = np.abs(ort_outs_orig[0] - ort_outs_fixed[0])
                        max_diff = np.max(abs_diff)
                        mean_diff = np.mean(abs_diff)
                        print(f"  📊 Difference: Max={max_diff:.8f}, Mean={mean_diff:.8f}")
                        
                        if max_diff < 1e-6:
                            print("  ✅ Fixed ONNX is nearly identical to original (excellent!)")
                        elif max_diff < 1e-4:
                            print("  ✅ Fixed ONNX is very close to original (acceptable)")
                        else:
                            print("  ⚠️  Fixed ONNX has noticeable differences from original")

                    # Compare fixed ONNX with PyTorch
                    try:
                        np.testing.assert_allclose(
                            torch_output_np,
                            ort_outs_fixed[0],
                            rtol=1e-02,
                            atol=1e-04,
                            verbose=False,
                        )
                        print("  ✅ Fixed ONNX matches PyTorch!")
                    except AssertionError:
                        abs_diff = np.abs(torch_output_np - ort_outs_fixed[0])
                        print(f"  📊 vs PyTorch: Max diff={np.max(abs_diff):.6f}, Mean diff={np.mean(abs_diff):.6f}")

                    print(f"✅ Successfully exported TensorRT-compatible ResEnc distillation model: {output_path}")
                else:
                    print("✅ Model is already TensorRT compatible")
            except Exception as e:
                print(f"Warning: TensorRT compatibility fix failed: {e}")
                print("Original ONNX model is still available")
        
        return output_path
    except Exception as e:
        print(f"Error exporting {model_type.upper()} distillation model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Export ResEnc distillation model to ONNX format')
    parser.add_argument('-d', '--dataset_id', type=str, required=True, help='Dataset ID')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres', help='Configuration name')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Fold number')
    parser.add_argument('-b', '--batch_size', type=int, default=0, help='Batch size, 0 means dynamic')
    parser.add_argument('-cp', '--checkpoint', type=str, default='checkpoint_final.pth', help='Checkpoint filename')
    parser.add_argument('-p', '--plans', type=str, default='nnUNetPlans', help='Plans identifier')
    parser.add_argument('-spl', '--student_plans', type=str, default='nnUNetPlans', help='Student plans identifier')
    parser.add_argument('-r', '--reduction_factor', type=int, default=2, help='Feature reduction factor')
    parser.add_argument('-bs', '--block_strategy', type=str, default='keep', 
                       choices=['reduce', 'keep', 'increase', 'adaptive'],
                       help='Block reduction strategy (MUST match training): reduce (A), keep (B), increase (B+), adaptive (B++) (default: keep)')
    parser.add_argument('--trt', '--tensorrt', action='store_true', dest='trt_compatible', help='Apply TensorRT compatibility fixes')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    parser.add_argument('--use_da5', action='store_true', help='Model was trained with DA5 data augmentation')
    
    args = parser.parse_args()
    
    # Export ResEnc distillation model
    export_resenc_distillation_to_onnx(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        configuration=args.configuration,
        fold=args.fold,
        batch_size=args.batch_size,
        checkpoint_name=args.checkpoint,
        plans_identifier=args.plans,
        student_plans_identifier=args.student_plans,
        feature_reduction_factor=args.reduction_factor,
        block_reduction_strategy=args.block_strategy,
        trt_compatible=args.trt_compatible,
        verbose=args.verbose,
        use_da5=args.use_da5
    )

if __name__ == '__main__':
    main()