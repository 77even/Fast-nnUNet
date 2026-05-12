#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

## Title: Fast-nnUNet — Primus distillation training entry point
## Authors: Justin Lee
## Description: CLI driver for distilling an upstream nnunetv2 Primus teacher into a smaller
##              Primus student. Mirrors the standard / ResEnc distillation entry points and
##              adds a `-ts/--teacher_size` flag to pick S / B / M / L.

import argparse
import json
import os
import sys

import torch
from batchgenerators.utilities.file_and_folder_operations import join

# Make sure the vendored nnunetv2 (on main) and our trainer module are importable
nnunet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nnunet_dir)

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from primus_distillation_trainer import (
    PRIMUS_TEACHER_SPECS,
    PRIMUS_TEACHER_TRAINER_CLASS,
    nnUNetDistillationPrimusTrainer,
    nnUNetDistillationPrimusTrainerDA5,
)


def get_dataset_name_from_id(dataset_id):
    try:
        dataset_id = int(dataset_id)
    except ValueError:
        return dataset_id
    for dataset_dir in os.listdir(nnUNet_raw):
        if dataset_dir.startswith(f"Dataset{dataset_id:03d}_"):
            return dataset_dir
    default_name = f"Dataset{dataset_id:03d}"
    print(
        f"Warning: No dataset directory with ID {dataset_id} found in {nnUNet_raw}, "
        f"using default name {default_name}"
    )
    return default_name


def run_primus_distillation_training(
    dataset_id,
    configuration="3d_fullres",
    fold=0,
    teacher_size="M",
    teacher_model_folder=None,
    teacher_folds=None,
    teacher_checkpoint_name="checkpoint_final.pth",
    teacher_plans_identifier="nnUNetPlans",
    alpha=0.3,
    temperature=3.0,
    feature_reduction_factor=2,
    warmup_duration_whole_net=50,
    continue_training=False,
    val_with_mirroring=True,
    rotate_training_folds=False,
    rotate_folds_frequency=5,
    device=None,
    epochs=1000,
    use_da5=False,
):
    """Run Primus knowledge-distillation training.

    Args mostly mirror ``fast_nnunet_resenc_distillation_train.run_resenc_distillation_training``.
    Primus-specific extras:
        teacher_size: One of "S", "B", "M", "L". Picks the upstream Primus teacher trainer.
        warmup_duration_whole_net: Number of epochs of linear LR warmup (matches upstream
            AbstractPrimus default of 50).
    """
    dataset_name = get_dataset_name_from_id(dataset_id)

    # Default teacher folder from teacher_size
    if teacher_model_folder is None:
        teacher_trainer_cls = PRIMUS_TEACHER_TRAINER_CLASS[teacher_size]
        teacher_model_folder = join(
            nnUNet_results,
            dataset_name,
            f"{teacher_trainer_cls}__{teacher_plans_identifier}__{configuration}",
        )
        print(
            f"Teacher model folder not specified, using upstream Primus default for size "
            f"{teacher_size}: {teacher_model_folder}"
        )

    if not os.path.exists(teacher_model_folder):
        raise FileNotFoundError(
            f"Primus teacher model folder does not exist: {teacher_model_folder}"
        )

    # Auto-detect teacher folds if not specified
    if teacher_folds is None:
        teacher_folds = []
        for item in os.listdir(teacher_model_folder):
            if os.path.isdir(join(teacher_model_folder, item)) and item.startswith("fold_"):
                try:
                    fold_num = int(item.split("_")[1])
                    if os.path.exists(join(teacher_model_folder, item, teacher_checkpoint_name)):
                        teacher_folds.append(fold_num)
                except ValueError:
                    continue
        teacher_folds.sort()
        if not teacher_folds:
            print("Warning: No available Primus teacher folds found, falling back to fold_0")
            teacher_folds = [0]
        else:
            print(f"Detected {len(teacher_folds)} available Primus teacher folds: {teacher_folds}")
    else:
        for tf in teacher_folds:
            cp = join(teacher_model_folder, f"fold_{tf}", teacher_checkpoint_name)
            if not os.path.exists(cp):
                raise FileNotFoundError(f"Primus teacher checkpoint missing: {cp}")

    dataset_json_file = join(nnUNet_raw, dataset_name, "dataset.json")
    if not os.path.exists(dataset_json_file):
        raise FileNotFoundError(f"Dataset json file does not exist: {dataset_json_file}")

    student_plans_file = join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json")
    if not os.path.exists(student_plans_file):
        raise FileNotFoundError(
            f"Student plans file does not exist: {student_plans_file}, "
            "make sure preprocessing for the configuration has been run"
        )

    with open(student_plans_file, "r") as f:
        plans = json.load(f)
    with open(dataset_json_file, "r") as f:
        dataset_json = json.load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    trainer_cls = (
        nnUNetDistillationPrimusTrainerDA5 if use_da5 else nnUNetDistillationPrimusTrainer
    )
    trainer = trainer_cls(
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
        teacher_size=teacher_size,
        warmup_duration_whole_net=warmup_duration_whole_net,
        device=device,
    )

    teacher_spec = PRIMUS_TEACHER_SPECS[teacher_size]
    print(f"\n============ Primus Knowledge Distillation Training Configuration ============")
    print(f"Dataset name: {dataset_name}")
    print(f"Configuration: {configuration}")
    print(f"Training fold: {fold}")
    print(
        f"Teacher Primus size: {teacher_size} "
        f"(embed_dim={teacher_spec['embed_dim']}, depth={teacher_spec['depth']}, "
        f"num_heads={teacher_spec['num_heads']})"
    )
    print(f"Teacher model folder: {teacher_model_folder}")
    print(f"Teacher model folds: {teacher_folds}")
    print(f"Teacher model checkpoint: {teacher_checkpoint_name}")
    print(f"Distillation loss weight alpha: {alpha}")
    print(f"Distillation temperature: {temperature}")
    print(f"Feature reduction factor (embed_dim/depth/heads divisor): {feature_reduction_factor}")
    print(f"Warmup duration (epochs): {warmup_duration_whole_net}")
    print(f"Continue training: {continue_training}")
    print(f"Validation with mirroring: {val_with_mirroring}")
    if rotate_training_folds:
        print(f"Rotating folds every {rotate_folds_frequency} epochs")
    print(f"Device: {device}")
    print(f"Maximum training epochs: {epochs}")
    print(f"================================================================================\n")

    trainer.num_epochs = epochs
    if not val_with_mirroring:
        trainer.inference_allowed_mirroring_axes = []

    expected_checkpoint_file = join(trainer.output_folder, "checkpoint_latest.pth")
    checkpoint_exists = os.path.exists(expected_checkpoint_file)

    if continue_training and checkpoint_exists:
        print(f"Continuing previous training, loading checkpoint: {expected_checkpoint_file}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not trainer.was_initialized:
            trainer.initialize()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trainer.load_student_checkpoint(expected_checkpoint_file)
        print(f"Loaded checkpoint, resuming from epoch {trainer.current_epoch}")
    else:
        if continue_training and not checkpoint_exists:
            print(
                f"Warning: could not find checkpoint {expected_checkpoint_file}, "
                "starting from scratch"
            )
        if not trainer.was_initialized:
            trainer.initialize()

    trainer.run_training()
    trainer.perform_actual_validation(False)


def main():
    parser = argparse.ArgumentParser(
        description="nnUNetv2 Primus knowledge-distillation training"
    )
    parser.add_argument("-d", "--dataset_id", type=str, required=True, help="Dataset ID (e.g., 793)")
    parser.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="3d_fullres",
        help="nnUNet configuration. Primus is 3D-only; patch_size must be divisible by 8 "
        "(default: 3d_fullres)",
    )
    parser.add_argument("-f", "--fold", type=int, default=0, help="Fold number for training (default: 0)")
    parser.add_argument(
        "-ts",
        "--teacher_size",
        type=str,
        default="M",
        choices=list(PRIMUS_TEACHER_SPECS),
        help="Upstream Primus teacher size: S / B / M / L (default: M)",
    )
    parser.add_argument(
        "-t",
        "--teacher_model_folder",
        type=str,
        help="Custom Primus teacher folder (auto-derived from --teacher_size if omitted)",
    )
    parser.add_argument(
        "-tf",
        "--teacher_folds",
        type=int,
        nargs="+",
        help="List of teacher fold numbers, e.g. 0 or 0 1 2 3 4. Default: auto-detect all available folds",
    )
    parser.add_argument(
        "-tcp",
        "--teacher_checkpoint",
        type=str,
        default="checkpoint_final.pth",
        help="Teacher model checkpoint filename (default: checkpoint_final.pth)",
    )
    parser.add_argument(
        "-tpl",
        "--teacher_plans",
        type=str,
        default="nnUNetPlans",
        help="Teacher plans identifier (default: nnUNetPlans)",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.3, help="Distillation loss weight (default: 0.3)")
    parser.add_argument("-temp", "--temperature", type=float, default=3.0, help="Distillation temperature (default: 3.0)")
    parser.add_argument(
        "-r",
        "--reduction_factor",
        type=int,
        default=2,
        help="Reduction factor applied to embed_dim/depth/num_heads (default: 2)",
    )
    parser.add_argument(
        "-w",
        "--warmup_epochs",
        type=int,
        default=50,
        help="Linear LR warmup duration in epochs (default: 50, matches upstream Primus)",
    )
    parser.add_argument("-d_device", "--device", type=str, help="Device to use, e.g., \"cuda:0\"")
    parser.add_argument("-c_continue", "--continue_training", action="store_true", help="Continue previous training")
    parser.add_argument(
        "-disable_mirroring",
        "--disable_val_mirroring",
        action="store_true",
        help="Disable mirroring during validation",
    )
    parser.add_argument(
        "-rotate_folds",
        "--rotate_training_folds",
        action="store_true",
        help="Rotate training folds periodically",
    )
    parser.add_argument(
        "-rotate_freq",
        "--rotate_folds_frequency",
        type=int,
        default=5,
        help="How often to rotate folds (in epochs, default: 5)",
    )
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Maximum training epochs (default: 1000)")
    parser.add_argument(
        "--use_da5",
        action="store_true",
        help="Use DA5 strong data augmentation (recommended for small datasets)",
    )

    args = parser.parse_args()

    run_primus_distillation_training(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        teacher_size=args.teacher_size,
        teacher_model_folder=args.teacher_model_folder,
        teacher_folds=args.teacher_folds,
        teacher_checkpoint_name=args.teacher_checkpoint,
        teacher_plans_identifier=args.teacher_plans,
        alpha=args.alpha,
        temperature=args.temperature,
        feature_reduction_factor=args.reduction_factor,
        warmup_duration_whole_net=args.warmup_epochs,
        continue_training=args.continue_training,
        val_with_mirroring=not args.disable_val_mirroring,
        rotate_training_folds=args.rotate_training_folds,
        rotate_folds_frequency=args.rotate_folds_frequency,
        device=args.device,
        epochs=args.epochs,
        use_da5=args.use_da5,
    )


if __name__ == "__main__":
    main()
