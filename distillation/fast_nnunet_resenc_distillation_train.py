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
## Description: ResEnc Knowledge Distillation Based on nnUNet architecture

import argparse
import os
import sys
import torch
import json
import inspect
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# Ensure using nnunetv2 from current directory
nnunet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nnunet_dir)

# Import paths from nnunetv2
from distillation.nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed

# Import nnUNetDistillationTrainer directly
from distillation.nnunetv2.training.nnUNetTrainer.variants.nnUNetDistillationTrainer import nnUNetDistillationTrainer
from distillation.nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

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

def run_resenc_distillation_training(dataset_id, 
                                   configuration='3d_fullres',
                                   fold=0,
                                   teacher_model_folder=None,
                                   teacher_folds=None,
                                   teacher_checkpoint_name='checkpoint_final.pth',
                                   teacher_plans_identifier='nnUNetResEncUNetLPlans',  # ResEnc teacher model
                                   student_plans_identifier='nnUNetPlans',  # Student model plans: nnUNetPlans(default) or nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans(ResEnc)
                                   alpha=0.3,
                                   temperature=3.0,
                                   feature_reduction_factor=2,
                                   continue_training=False,
                                   val_with_mirroring=True,
                                   rotate_training_folds=False,
                                   rotate_folds_frequency=5,
                                   device=None,
                                   epochs=1000):
    """
    Run nnUNet ResEnc knowledge distillation training.
    
    Parameters:
        dataset_id: Dataset ID (e.g., 793)
        configuration: nnUNet configuration, e.g., '3d_fullres'
        fold: Fold number for training
        teacher_model_folder: Path to teacher model folder
        teacher_folds: List of teacher model fold numbers, e.g., [0] or [0,1,2,3,4], if None, auto-detect all available folds
        teacher_checkpoint_name: Teacher model checkpoint filename
        teacher_plans_identifier: Teacher model plans identifier (ResEnc)
        student_plans_identifier: Student model plans identifier (can be default or ResEnc)
        alpha: Distillation loss weight
        temperature: Distillation temperature
        feature_reduction_factor: Feature reduction factor
        continue_training: Whether to continue previous training
        val_with_mirroring: Whether to use mirroring during validation
        rotate_training_folds: Whether to rotate training folds during training
        rotate_folds_frequency: How often to rotate folds (in epochs)
        device: Device to use, e.g., 'cuda:0'
        epochs: Maximum number of training epochs
    """
    # Parse dataset ID to full name
    dataset_name = get_dataset_name_from_id(dataset_id)
    
    # If teacher_model_folder is None, auto-construct ResEnc teacher model path
    if teacher_model_folder is None:
        teacher_model_folder = join(nnUNet_results, dataset_name, f"nnUNetTrainer__{teacher_plans_identifier}__{configuration}")
        print(f"Teacher model folder not specified, using ResEnc default: {teacher_model_folder}")

    # Ensure necessary files exist
    if not os.path.exists(teacher_model_folder):
        raise FileNotFoundError(f"ResEnc teacher model folder does not exist: {teacher_model_folder}")
    
    # If teacher_folds is None, auto-detect all available folds
    if teacher_folds is None:
        teacher_folds = []
        # Find all fold_X folders
        for item in os.listdir(teacher_model_folder):
            if os.path.isdir(join(teacher_model_folder, item)) and item.startswith("fold_"):
                try:
                    fold_num = int(item.split("_")[1])
                    # Check if the fold is valid (contains checkpoint file)
                    if os.path.exists(join(teacher_model_folder, item, teacher_checkpoint_name)):
                        teacher_folds.append(fold_num)
                except ValueError:
                    continue
        
        # Sort folds in order
        teacher_folds.sort()
        
        if len(teacher_folds) == 0:
            print(f"Warning: No available ResEnc teacher model folds found, will use fold_0")
            teacher_folds = [0]
        else:
            print(f"Detected {len(teacher_folds)} available ResEnc teacher model folds: {teacher_folds}")
    else:
        # Check if all specified teacher_folds exist
        for teacher_fold in teacher_folds:
            teacher_checkpoint_path = join(teacher_model_folder, f"fold_{teacher_fold}", teacher_checkpoint_name)
            if not os.path.exists(teacher_checkpoint_path):
                raise FileNotFoundError(f"ResEnc teacher model checkpoint does not exist: {teacher_checkpoint_path}")
    
    dataset_json_file = join(nnUNet_raw, dataset_name, "dataset.json")
    if not os.path.exists(dataset_json_file):
        raise FileNotFoundError(f"Dataset json file does not exist: {dataset_json_file}")
    
    # Load student model plans and dataset_json
    student_plans_file = join(nnUNet_preprocessed, dataset_name, f"{student_plans_identifier}.json")
    if not os.path.exists(student_plans_file):
        raise FileNotFoundError(f"Student model plans file does not exist: {student_plans_file}, please ensure corresponding preprocessing step has been run")
        
    with open(student_plans_file, 'r') as f:
        plans = json.load(f)
        
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create distillation trainer instance directly
    trainer = nnUNetDistillationTrainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        teacher_model_folder=teacher_model_folder,
        teacher_fold=teacher_folds,
        teacher_checkpoint_name=teacher_checkpoint_name,
        alpha=alpha,
        temperature=temperature,
        feature_reduction_factor=feature_reduction_factor,
        rotate_training_folds=rotate_training_folds,
        rotate_folds_frequency=rotate_folds_frequency,
        device=device
    )
    
    # Output training configuration information
    print(f"\n============ ResEnc Knowledge Distillation Training Configuration ============")
    print(f"Dataset name: {dataset_name}")
    print(f"Configuration: {configuration}")
    print(f"Training fold: {fold}")
    print(f"Teacher model folder: {teacher_model_folder}")
    print(f"Teacher model plans: {teacher_plans_identifier}")
    print(f"Student model plans: {student_plans_identifier}")
    print(f"Teacher model folds: {teacher_folds}")
    print(f"Teacher model checkpoint: {teacher_checkpoint_name}")
    print(f"Distillation loss weight alpha: {alpha}")
    print(f"Distillation temperature: {temperature}")
    print(f"Feature reduction factor: {feature_reduction_factor}")
    print(f"Continue training: {continue_training}")
    print(f"Validation with mirroring: {val_with_mirroring}")
    print(f"Rotate training folds: {rotate_training_folds}")
    if rotate_training_folds:
        print(f"Rotate folds frequency: {rotate_folds_frequency} epochs")
    print(f"Device: {device}")
    print(f"Maximum training epochs: {epochs}")
    print(f"=============================================================================\n")
    
    # Set maximum epoch count
    trainer.num_epochs = epochs
    
    # Set whether to use mirroring for validation
    if not val_with_mirroring:
        trainer.inference_allowed_mirroring_axes = []
    
    # First check if checkpoint file exists
    expected_checkpoint_file = join(trainer.output_folder, "checkpoint_latest.pth")
    checkpoint_exists = os.path.exists(expected_checkpoint_file)
    
    # Determine training flow based on whether to continue training and if checkpoint exists
    if continue_training and checkpoint_exists:
        print(f"Continuing previous training, loading checkpoint: {expected_checkpoint_file}")
        
        # Add a precaution: don't initialize in the constructor, but actively initialize before reading the checkpoint file
        # This way there is no repeated initialization problem
        if not trainer.was_initialized:
            trainer.initialize()
            
        # Load checkpoint
        trainer.load_student_checkpoint(expected_checkpoint_file)
        print(f"Successfully loaded checkpoint, will continue training from epoch {trainer.current_epoch}")
    else:
        # If not continuing training or no checkpoint file, start training from scratch
        if continue_training and not checkpoint_exists:
            print(f"Warning: Could not find checkpoint file {expected_checkpoint_file}, will start training from scratch")
        
        # Initialize trainer
        if not trainer.was_initialized:
            trainer.initialize()
    
    # Run training
    trainer.run_training()

    # Run validation
    trainer.perform_actual_validation(False)

def main():
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='nnUNetv2 ResEnc Knowledge Distillation Training')
    parser.add_argument('-d', '--dataset_id', type=str, required=True, help='Dataset ID (e.g., 793)')
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres', help='nnUNet configuration (default: 3d_fullres)')
    parser.add_argument('-f', '--fold', type=int, default=0, help='Fold number for training (default: 0)')
    parser.add_argument('-t', '--teacher_model_folder', type=str, help='ResEnc teacher model folder path (if not provided, will be auto-constructed)')
    parser.add_argument('-tf', '--teacher_folds', type=int, nargs='+', 
                       help='List of teacher model fold numbers, e.g., 0 or 0 1 2 3 4. If not specified, all available folds will be auto-detected')
    parser.add_argument('-tcp', '--teacher_checkpoint', type=str, default='checkpoint_final.pth', 
                       help='Teacher model checkpoint filename (default: checkpoint_final.pth)')
    parser.add_argument('-tpl', '--teacher_plans', type=str, default='nnUNetResEncUNetLPlans',
                       help='Teacher model plans identifier (default: nnUNetResEncUNetLPlans)')
    parser.add_argument('-spl', '--student_plans', type=str, default='nnUNetPlans',
                       help='Student model plans identifier (default: nnUNetPlans)')
    parser.add_argument('-a', '--alpha', type=float, default=0.3, help='Distillation loss weight (default: 0.3)')
    parser.add_argument('-temp', '--temperature', type=float, default=3.0, help='Distillation temperature (default: 3.0)')
    parser.add_argument('-r', '--reduction_factor', type=int, default=2, help='Feature reduction factor (default: 2)')
    parser.add_argument('-d_device', '--device', type=str, help='Device to use, e.g., "cuda:0"')
    parser.add_argument('-c_continue', '--continue_training', action='store_true', help='Whether to continue previous training')
    parser.add_argument('-disable_mirroring', '--disable_val_mirroring', action='store_true', help='Disable mirroring during validation')
    parser.add_argument('-rotate_folds', '--rotate_training_folds', action='store_true', 
                       help='Enable rotating training folds periodically')
    parser.add_argument('-rotate_freq', '--rotate_folds_frequency', type=int, default=5, 
                       help='How often to rotate folds (in epochs) (default: 5)')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Maximum number of training epochs (default: 1000)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Run ResEnc distillation training
    run_resenc_distillation_training(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        teacher_model_folder=args.teacher_model_folder,
        teacher_folds=args.teacher_folds,
        teacher_checkpoint_name=args.teacher_checkpoint,
        teacher_plans_identifier=args.teacher_plans,
        student_plans_identifier=args.student_plans,
        alpha=args.alpha,
        temperature=args.temperature,
        feature_reduction_factor=args.reduction_factor,
        continue_training=args.continue_training,
        val_with_mirroring=not args.disable_val_mirroring,
        rotate_training_folds=args.rotate_training_folds,
        rotate_folds_frequency=args.rotate_folds_frequency,
        device=args.device,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()