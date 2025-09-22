#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Title: Fast-nnUNet: High-performance medical image segmentation framework based on the nnUNetv2 architecture
## Authors: Justin Lee
## Description: Convet nnUNet models to ONNX models

import os
import sys
import torch
import argparse
import json
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isfile

# Ensure using nnunetv2 from current directory
nnunet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nnunet_dir)

# Import paths and related functionalities from nnunetv2
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.nnUNetTrainer.variants.nnUNetDistillationTrainer import nnUNetDistillationTrainer, nnUNetDistillationTrainerDA5, LiteNNUNetStudent

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

def export_to_onnx(dataset_id,
                   configuration='3d_fullres',
                   fold=0,
                   feature_reduction_factor=2,
                   checkpoint_name='checkpoint_final.pth',
                   output_path=None,
                   device=None,
                   dynamic_axes=True,
                   input_shape=None,
                   nnunet_style=False,
                   verbose=False,
                   use_da5=False):
    """
    Export trained nnUNet distillation student model to ONNX format
    
    Parameters:
        dataset_id: Dataset ID (e.g., 776)
        configuration: nnUNet configuration, e.g., '3d_fullres'
        fold: Model training fold number
        feature_reduction_factor: Feature reduction factor, must match training settings
        checkpoint_name: Checkpoint filename
        output_path: Output path for ONNX file, if None will be auto-generated
        device: Device to use, e.g., 'cuda:0'
        dynamic_axes: Whether to use dynamic axes, if True allows different input shapes
        input_shape: Custom input shape (b, c, x, y, z), default None uses model's accepted shape
        nnunet_style: Whether to use nnUNet style output
        verbose: Whether to display detailed information
        use_da5: Whether the model was trained with DA5 data augmentation
    """
    # Parse dataset ID to full name
    dataset_name = get_dataset_name_from_id(dataset_id)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Determine model folder path based on whether DA5 was used
    if use_da5:
        trainer_name = "nnUNetDistillationTrainerDA5"
    else:
        trainer_name = "nnUNetDistillationTrainer"
    
    model_folder = join(nnUNet_results, dataset_name, f"{trainer_name}__nnUNetPlans__{configuration}")
    model_folder_fold = join(model_folder, f"fold_{fold}")
    
    if not os.path.exists(model_folder_fold):
        raise FileNotFoundError(f"Model folder does not exist: {model_folder_fold}")
    
    # Determine checkpoint file path
    checkpoint_path = join(model_folder_fold, checkpoint_name)
    if not isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    # Load plans and dataset_json
    plans_file = join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json")
    if not isfile(plans_file):
        raise FileNotFoundError(f"Plans file does not exist: {plans_file}")
    
    dataset_json_file = join(nnUNet_raw, dataset_name, "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(f"Dataset json file does not exist: {dataset_json_file}")
    
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)
    
    # Create trainer instance (just to load model parameters)
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
    
    # Initialize network architecture (full trainer initialization not needed)
    num_input_channels = determine_num_input_channels(trainer.plans_manager, trainer.configuration_manager, dataset_json)
    num_output_channels = trainer.label_manager.num_segmentation_heads

    # if nnunet_style is True, force using single channel input
    if nnunet_style:
        num_input_channels = 1
        print(f"Using single channel fixed size mode, force using 1 input channel")
    
    print(f"Number of input channels: {num_input_channels}, Number of output channels: {num_output_channels}")
    
    # Get network architecture parameters
    if configuration in plans['configurations']:
        config = plans['configurations'][configuration]
        
        # Check keys in plans to accommodate different versions of nnUNet
        if verbose:
            print("Keys in configuration:", list(config.keys()))
        
        # Get architecture information
        if 'architecture' in config:
            arch_info = config['architecture']
            if verbose:
                print(f"Found architecture info: {arch_info}")
                
            # Extract parameters from architecture
            if isinstance(arch_info, dict) and 'arch_kwargs' in arch_info:
                arch_kwargs = arch_info['arch_kwargs']
                n_stages = arch_kwargs.get('n_stages', 6)
                features_per_stage = arch_kwargs.get('features_per_stage', [32, 64, 128, 256, 320, 320][:n_stages])
                strides = arch_kwargs.get('strides', [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1))
                kernel_sizes = arch_kwargs.get('kernel_sizes', [[3, 3, 3]] * n_stages)
                
                if verbose:
                    print(f"Extracted from architecture: n_stages={n_stages}, features={features_per_stage}")
            else:
                # Fall back to defaults
                n_stages = 6
                features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
                strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
                kernel_sizes = [[3, 3, 3]] * n_stages
        else:
            # Get number of stages and features - handle different plans formats
            if 'pool_op_kernel_sizes' in config:
                n_stages = len(config['pool_op_kernel_sizes'])
                features_per_stage = config.get('features_per_stage', [32, 64, 128, 256, 320, 320][:n_stages])
                strides = config['pool_op_kernel_sizes']
                kernel_sizes = config.get('conv_kernel_sizes', [[3, 3, 3]] * n_stages)
            elif 'architecture_kwargs' in config and 'arch_kwargs' in config['architecture_kwargs']:
                arch_kwargs = config['architecture_kwargs']['arch_kwargs']
                n_stages = arch_kwargs.get('n_stages', 6)
                features_per_stage = arch_kwargs.get('features_per_stage', [32, 64, 128, 256, 320, 320][:n_stages])
                strides = arch_kwargs.get('strides', [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1))
                kernel_sizes = arch_kwargs.get('kernel_sizes', [[3, 3, 3]] * n_stages)
            else:
                # If configuration not found, use default values
                print("Warning: Cannot find complete network architecture configuration in plans, using defaults")
                n_stages = 6
                features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
                strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
                kernel_sizes = [[3, 3, 3]] * n_stages
        
        # Apply feature reduction factor
        lite_features_per_stage = [max(f // feature_reduction_factor, 8) for f in features_per_stage]
        
        if verbose:
            print(f"Number of stages: {n_stages}")
            print(f"Original features: {features_per_stage}")
            print(f"Reduced features: {lite_features_per_stage}")
            print(f"Strides: {strides}")
            print(f"Convolution kernel sizes: {kernel_sizes}")
    else:
        raise ValueError(f"Configuration {configuration} does not exist in plans")
    
    # Load model parameters
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
    
    # Create student model
    # Get network structure parameters from checkpoint
    if 'network_weights' in checkpoint:
        # Handle storage using network_weights
        state_dict = checkpoint['network_weights']
        
        # Build model
        model = LiteNNUNetStudent(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=n_stages,
            features_per_stage=lite_features_per_stage,
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[tuple(k) if isinstance(k, list) else k for k in kernel_sizes],
            strides=[tuple(p) if isinstance(p, list) else p for p in strides],
            n_conv_per_stage=[2] * n_stages,  # Default 2 convolution layers per stage
            n_conv_per_stage_decoder=[2] * (n_stages - 1),
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False  # Don't use deep supervision for ONNX export
        )
    elif 'network_config' in checkpoint and 'network' in checkpoint['network_config']:
        network_config = checkpoint['network_config']['network']
        model = LiteNNUNetStudent(**network_config)
    elif 'state_dict' in checkpoint:
        # Handle storage using state_dict
        state_dict = checkpoint['state_dict']
        
        # Build model
        model = LiteNNUNetStudent(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=n_stages,
            features_per_stage=lite_features_per_stage,
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[tuple(k) if isinstance(k, list) else k for k in kernel_sizes],
            strides=[tuple(p) if isinstance(p, list) else p for p in strides],
            n_conv_per_stage=[2] * n_stages,  # Default 2 convolution layers per stage
            n_conv_per_stage_decoder=[2] * (n_stages - 1),
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False  # Don't use deep supervision for ONNX export
        )
    else:
        # If no network configuration in checkpoint, build manually
        # Build model
        model = LiteNNUNetStudent(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=n_stages,
            features_per_stage=lite_features_per_stage,
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[tuple(k) if isinstance(k, list) else k for k in kernel_sizes],
            strides=[tuple(p) if isinstance(p, list) else p for p in strides],
            n_conv_per_stage=[2] * n_stages,  # Default 2 convolution layers per stage
            n_conv_per_stage_decoder=[2] * (n_stages - 1),
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False  # Don't use deep supervision for ONNX export
        )
    
    # Load network parameters
    if 'network_weights' in checkpoint:
        # Handle potential 'module.' prefix (for DataParallel wrapped models)
        state_dict = checkpoint['network_weights']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Get available parameter keys in the model
        model_keys = set(model.state_dict().keys())
        # Filter parameters that exist in the model
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
        
        # Load parameters
        model.load_state_dict(filtered_state_dict, strict=False)
        
        if verbose:
            missing_keys = model_keys - set(filtered_state_dict.keys())
            if missing_keys:
                print(f"Warning: The following parameters do not exist in checkpoint: {missing_keys}")
    elif 'state_dict' in checkpoint:
        # Handle potential 'module.' prefix (for DataParallel wrapped models)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Only load network parameters (excluding deep supervision)
        # Get available parameter keys in the model
        model_keys = set(model.state_dict().keys())
        # Filter parameters that exist in the model
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
        
        # Load parameters
        model.load_state_dict(filtered_state_dict, strict=False)
        
        if verbose:
            missing_keys = model_keys - set(filtered_state_dict.keys())
            if missing_keys:
                print(f"Warning: The following parameters do not exist in checkpoint: {missing_keys}")
    else:
        raise ValueError(f"Checkpoint file {checkpoint_path} does not contain usable weights ('state_dict' or 'network_weights')")
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Prepare export path
    if output_path is None:
        output_dir = join(model_folder_fold, "exported_models")
        os.makedirs(output_dir, exist_ok=True)

        # add special identifier for single channel fixed size model
        da5_suffix = "_da5" if use_da5 else ""
        if nnunet_style:
            output_path = join(output_dir, f"model_fold{fold}_{feature_reduction_factor}x{da5_suffix}_nnunet_format.onnx")
        else:
            output_path = join(output_dir, f"model_fold{fold}_{feature_reduction_factor}x{da5_suffix}.onnx")
    
    # Create input sample - use typical 3D medical image shape or custom shape
    if input_shape is None:
        # Default size estimation (assuming input patch_size is the same as during training)
        if 'patch_size' in config:
            patch_size = config['patch_size']
            input_shape = (1, num_input_channels, *patch_size)
        else:
            # Get patch_size from trainer
            try:
                patch_size = trainer.configuration_manager.patch_size
                input_shape = (1, num_input_channels, *patch_size)
            except:
                # Default shape
                input_shape = (1, num_input_channels, 128, 128, 128)

    # if nnunet_style is True, force using single channel fixed size input
    if nnunet_style:
        # force using single channel and fixed size
        if input_shape is None or len(input_shape) != 5:
            # use default fixed size
            input_shape = (1, 1, 128, 128, 128)
        else:
            # keep original spatial dimensions, but set channel number to 1
            input_shape = (input_shape[0], 1, input_shape[2], input_shape[3], input_shape[4])
        
        print(f"Using single channel fixed size mode, input shape: {input_shape}")
    
    dummy_input = torch.zeros(input_shape, dtype=torch.float32).to(device)
    
    print(f"Exporting model with input shape {input_shape}")
    
    # Set dynamic axes for ONNX export
    if dynamic_axes and not nnunet_style:
        # Batch size and spatial dimensions are dynamic
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width', 4: 'depth'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width', 4: 'depth'}
        }
    elif dynamic_axes and nnunet_style:
        # only batch size is dynamic, spatial dimensions are fixed
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model,                       # Model instance
            dummy_input,                 # Model input
            output_path,                 # Output file
            export_params=True,          # Store trained parameter weights
            opset_version=17,            # ONNX operator set version
            do_constant_folding=True,    # Constant folding optimization
            input_names=['input'],       # Input names
            output_names=['output'],     # Output names
            dynamic_axes=dynamic_axes_dict,  # Dynamic dimensions
            verbose=verbose              # Verbose output
        )
        print(f"Model successfully exported to {output_path}")
        
        # Try to simplify model (if onnx and onnxsim are installed)
        try:
            import onnx
            from onnxsim import simplify
            
            print("Attempting to simplify ONNX model...")
            model_onnx = onnx.load(output_path)
            model_simp, check = simplify(model_onnx)
            
            if check:
                onnx.save(model_simp, output_path)
                print(f"Simplified model saved to {output_path}")
            else:
                print("Model simplification failed, keeping original exported model")
        except ImportError:
            print("Note: onnx or onnxsim not installed, skipping model simplification step")
            print("Tip: Use 'pip install onnx onnxsim' to install required libraries")
        
        return output_path
    except Exception as e:
        print(f"Error exporting model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Export nnUNet distillation student model to ONNX format')
    parser.add_argument('-d', '--dataset_id', type=str, required=True, help='Dataset ID (e.g., 776)')
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres', help='nnUNet configuration (default: 3d_fullres)')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Model training fold number (default: 0)')
    parser.add_argument('-r', '--reduction_factor', type=int, default=2, help='Feature reduction factor (default: 2)')
    parser.add_argument('-cp', '--checkpoint', type=str, default='checkpoint_final.pth', help='Checkpoint filename (default: checkpoint_final.pth)')
    parser.add_argument('-o', '--output', type=str, help='Output ONNX file path, if not specified will be auto-generated')
    parser.add_argument('-d_device', '--device', type=str, help='Device to use, e.g., "cuda:0"')
    parser.add_argument('--static', action='store_false', dest='dynamic_axes', help='Use static input shape instead of dynamic shape')
    parser.add_argument('--input_shape', type=int, nargs='+', help='Custom input shape (b c x y z)')
    parser.add_argument('--nnunet_format', action='store_true', dest='single_channel_fixed_size', help='导出单通道固定尺寸模型 [batch_size, 1, 固定尺寸]')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display detailed information')
    parser.add_argument('--use_da5', action='store_true', help='Model was trained with DA5 data augmentation')
    
    args = parser.parse_args()
    
    # Process input shape parameter
    input_shape = None
    if args.input_shape is not None:
        if len(args.input_shape) != 5:
            parser.error("Input shape must be 5 integers: b c x y z")
        input_shape = tuple(args.input_shape)
    
    # Export model
    export_to_onnx(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        feature_reduction_factor=args.reduction_factor,
        checkpoint_name=args.checkpoint,
        output_path=args.output,
        device=args.device,
        dynamic_axes=args.dynamic_axes,
        input_shape=input_shape,
        nnunet_style=args.nnunet_format,
        verbose=args.verbose,
        use_da5=args.use_da5
    )

if __name__ == '__main__':
    main() 
