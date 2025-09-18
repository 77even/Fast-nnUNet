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
from nnunetv2.training.nnUNetTrainer.variants.nnUNetDistillationTrainer import nnUNetDistillationTrainer, LiteNNUNetStudent
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
    dataset_json_path = join(nnUNet_raw, dataset_name, "dataset.json")
    if isfile(dataset_json_path):
        output_json_path = join(output_dir, "dataset.json")
        copy_file(dataset_json_path, output_json_path)
        print(f"Dataset json file exported to: {output_json_path}")
    else:
        print(f"Warning: Dataset json file not found: {dataset_json_path}")

def load_model_from_checkpoint(checkpoint_path, plans, dataset_json, configuration, fold, 
                             student_plans_identifier, feature_reduction_factor, device, verbose=False):
    """Load ResEnc distillation student model from checkpoint"""
    
    # Create trainer instance to get model architecture parameters
    trainer = nnUNetDistillationTrainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        teacher_model_folder=None,
        feature_reduction_factor=feature_reduction_factor,
        device=device
    )
    
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
            else:
                # Fall back to defaults
                n_stages = 6
                features_per_stage = [32, 64, 128, 256, 320, 320][:n_stages]
                strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
                kernel_sizes = [[3, 3, 3]] * n_stages
        else:
            # Get from other configuration keys
            if 'pool_op_kernel_sizes' in config:
                n_stages = len(config['pool_op_kernel_sizes'])
                features_per_stage = config.get('features_per_stage', [32, 64, 128, 256, 320, 320][:n_stages])
                strides = config['pool_op_kernel_sizes']
                kernel_sizes = config.get('conv_kernel_sizes', [[3, 3, 3]] * n_stages)
            else:
                # Use default values
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
    else:
        raise ValueError(f"Configuration {configuration} does not exist in plans")
    
    # Create student model
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
        deep_supervision=False  # Don't use deep supervision for ONNX export
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
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    if verbose:
        missing_keys = model_keys - set(filtered_state_dict.keys())
        if missing_keys:
            print(f"Warning: The following parameters do not exist in checkpoint: {missing_keys}")
    
    model.eval()
    model.to(device)
    
    return model, num_input_channels, num_output_channels, config

def export_resenc_distillation_to_onnx(dataset_id,
                                     output_dir=None,
                                     configuration='3d_fullres',
                                     fold=0,
                                     batch_size=0,
                                     checkpoint_name='checkpoint_final.pth',
                                     plans_identifier='nnUNetPlans',
                                     student_plans_identifier='nnUNetPlans',
                                     feature_reduction_factor=2,
                                     verbose=False):
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
        verbose: Show detailed output
    """
    # Parse dataset ID to full name
    dataset_name = get_dataset_name_from_id(dataset_id)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model folder path
    model_folder = join(nnUNet_results, dataset_name, f"nnUNetDistillationTrainer__{student_plans_identifier}__{configuration}")
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
    model, num_input_channels, num_output_channels, config = load_model_from_checkpoint(
        checkpoint_path, plans, dataset_json, configuration, fold,
        student_plans_identifier, feature_reduction_factor, device, verbose
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
    
    # Create dummy input
    if batch_size > 0:
        # Fixed batch size
        dummy_input = torch.zeros((batch_size, num_input_channels, *input_shape[2:]), dtype=torch.float32).to(device)
        dynamic_axes = None
        onnx_filename = f"resenc_distillation_fold{fold}_batch{batch_size}_r{feature_reduction_factor}.onnx"
    else:
        # Dynamic batch size
        dummy_input = torch.zeros(input_shape, dtype=torch.float32).to(device)
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        onnx_filename = f"resenc_distillation_fold{fold}_dynamic_r{feature_reduction_factor}.onnx"
    
    output_path = join(output_dir, onnx_filename)
    
    print(f"Exporting ResEnc distillation model with input shape {dummy_input.shape}")
    
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
            dynamic_axes=dynamic_axes,   # Dynamic dimensions
            verbose=verbose              # Verbose output
        )
        print(f"ResEnc distillation model successfully exported to {output_path}")
        
        # Export dataset json
        export_dataset_json(dataset_name, output_dir)
        
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
        print(f"Error exporting ResEnc distillation model: {e}")
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
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
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
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()