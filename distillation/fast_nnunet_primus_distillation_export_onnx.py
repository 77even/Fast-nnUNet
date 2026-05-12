#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

## Title: Fast-nnUNet — Primus distillation ONNX export entry point
## Authors: Justin Lee
## Description: Export a distilled Primus student checkpoint to ONNX.

import argparse
import json
import os
import sys

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join

nnunet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nnunet_dir)

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from primus_distillation_trainer import (
    PRIMUS_TEACHER_SPECS,
    LitePrimusStudent,
    reduce_primus_dims,
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


def export_primus_distillation_to_onnx(
    dataset_id,
    configuration="3d_fullres",
    fold=0,
    teacher_size="M",
    feature_reduction_factor=2,
    checkpoint_name="checkpoint_final.pth",
    output_path=None,
    device=None,
    dynamic_axes=True,
    input_shape=None,
    simplify_onnx=False,
    verbose=False,
    use_da5=False,
):
    """Export a distilled Primus student model to ONNX.

    Args:
        dataset_id: Dataset ID, e.g. 793.
        configuration: nnUNet configuration. Primus is 3D-only.
        fold: Student model fold to export.
        teacher_size: Teacher Primus size used during training (S/B/M/L). Drives the student
            architecture reconstruction so the checkpoint weights load.
        feature_reduction_factor: Same value used during training.
        checkpoint_name: Filename inside the fold folder.
        output_path: Custom .onnx path. Auto-named if None.
        device: Torch device. Defaults to CUDA if available, else CPU.
        dynamic_axes: If True, mark batch + spatial axes as dynamic.
        input_shape: Custom dummy input shape; defaults to (1, C, *patch_size).
        simplify_onnx: Run onnx-simplifier afterwards if installed.
        verbose: Extra logging.
        use_da5: Set if the checkpoint was trained with DA5 (only affects the trainer name
            used for the folder path).
    """
    if teacher_size not in PRIMUS_TEACHER_SPECS:
        raise ValueError(
            f"teacher_size must be one of {sorted(PRIMUS_TEACHER_SPECS)}; got {teacher_size!r}"
        )

    dataset_name = get_dataset_name_from_id(dataset_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    trainer_name = (
        "nnUNetDistillationPrimusTrainerDA5" if use_da5 else "nnUNetDistillationPrimusTrainer"
    )
    model_folder = join(
        nnUNet_results, dataset_name, f"{trainer_name}__nnUNetPlans__{configuration}"
    )
    model_folder_fold = join(model_folder, f"fold_{fold}")
    if not os.path.exists(model_folder_fold):
        raise FileNotFoundError(f"Model folder does not exist: {model_folder_fold}")

    checkpoint_path = join(model_folder_fold, checkpoint_name)
    if not isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    plans_file = join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json")
    if not isfile(plans_file):
        raise FileNotFoundError(f"Plans file does not exist: {plans_file}")
    dataset_json_file = join(nnUNet_raw, dataset_name, "dataset.json")
    if not isfile(dataset_json_file):
        raise FileNotFoundError(f"Dataset json file does not exist: {dataset_json_file}")

    with open(plans_file, "r") as f:
        plans = json.load(f)
    with open(dataset_json_file, "r") as f:
        dataset_json = json.load(f)

    plans_manager = PlansManager(plans)
    cfg_manager = plans_manager.get_configuration(configuration)
    patch_size = tuple(cfg_manager.patch_size)
    if len(patch_size) != 3:
        raise ValueError(f"Primus is 3D-only; got patch_size {patch_size}")
    if any(p % 8 != 0 for p in patch_size):
        raise ValueError(
            f"Primus requires patch_size divisible by 8; got {patch_size}"
        )

    num_input_channels = determine_num_input_channels(plans_manager, cfg_manager, dataset_json)
    num_output_channels = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
    print(
        f"Configuration: {configuration}. patch_size={patch_size}. "
        f"Input channels: {num_input_channels}, output channels: {num_output_channels}"
    )

    # Reconstruct student architecture from teacher_size + reduction_factor
    spec = PRIMUS_TEACHER_SPECS[teacher_size]
    s_embed, s_depth, s_heads = reduce_primus_dims(
        spec["embed_dim"], spec["depth"], spec["num_heads"], feature_reduction_factor
    )
    print(
        f"Primus student (teacher {teacher_size}, r={feature_reduction_factor}): "
        f"embed_dim={s_embed}, depth={s_depth}, num_heads={s_heads}"
    )

    model = LitePrimusStudent(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        embed_dim=s_embed,
        depth=s_depth,
        num_heads=s_heads,
        patch_size=patch_size,
    )

    # Load checkpoint
    try:
        import numpy.core.multiarray as _np_core
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([_np_core.scalar])
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "network_weights" in checkpoint:
        state_dict = checkpoint["network_weights"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains neither 'network_weights' nor 'state_dict'"
        )

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[len("module."):]] = v if k.startswith("module.") else cleaned.setdefault(k, v)
    # Simpler: strip `module.` and load with strict=False
    cleaned = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    load_result = model.load_state_dict(cleaned, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        if verbose:
            print(
                f"Load issues — missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)}"
            )

    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params (~{total_params * 4 / 1024 / 1024:.1f} MB)")

    if output_path is None:
        output_dir = join(model_folder_fold, "exported_models")
        os.makedirs(output_dir, exist_ok=True)
        da5_suffix = "_da5" if use_da5 else ""
        output_path = join(
            output_dir,
            f"primus_{teacher_size}_{configuration}_fold{fold}_r{feature_reduction_factor}{da5_suffix}.onnx",
        )

    if input_shape is None:
        input_shape = (1, num_input_channels, *patch_size)
    torch.manual_seed(42)
    dummy_input = torch.randn(input_shape, dtype=torch.float32, device=device)

    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
            "output": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
        }
    else:
        dynamic_axes_dict = None

    print(f"Exporting Primus student to ONNX (input: {tuple(dummy_input.shape)})...")
    with torch.no_grad():
        torch_output = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=verbose,
    )
    print(f"Exported to: {output_path}")

    # Validate
    try:
        import onnx
        from onnx import checker
        from onnxruntime import InferenceSession

        onnx_model = onnx.load(output_path)
        checker.check_model(onnx_model)

        ort_session = InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        torch_output_np = torch_output.detach().cpu().numpy()
        abs_diff = np.abs(torch_output_np - ort_outputs[0])
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        if max_diff < 0.01:
            print(f"   Excellent match (max={max_diff:.6f}, mean={mean_diff:.6f})")
        elif max_diff < 0.5:
            print(f"   Good match (max={max_diff:.6f}, mean={mean_diff:.6f})")
        else:
            print(f"   Difference detected (max={max_diff:.6f}, mean={mean_diff:.6f})")
    except ImportError as exc:
        print(f"Skipping ONNX validation ({exc}); install with `pip install -e '.[onnx]'`.")

    if simplify_onnx:
        try:
            from onnxsim import simplify  # type: ignore
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified, ok = simplify(onnx_model)
            if ok:
                onnx.save(simplified, output_path)
                print(f"Simplified ONNX written to: {output_path}")
            else:
                print("onnx-simplifier reported failure; keeping original.")
        except ImportError:
            print("onnx-simplifier not installed; skipping --simplify path.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export distilled Primus student to ONNX")
    parser.add_argument("-d", "--dataset_id", type=str, required=True, help="Dataset ID")
    parser.add_argument(
        "-c",
        "--configuration",
        type=str,
        default="3d_fullres",
        help="nnUNet configuration (Primus is 3D-only) (default: 3d_fullres)",
    )
    parser.add_argument("-f", "--fold", type=int, default=0, help="Fold number (default: 0)")
    parser.add_argument(
        "-ts",
        "--teacher_size",
        type=str,
        default="M",
        choices=list(PRIMUS_TEACHER_SPECS),
        help="Primus teacher size used during training (default: M)",
    )
    parser.add_argument(
        "-r", "--reduction_factor", type=int, default=2, help="Reduction factor used during training (default: 2)"
    )
    parser.add_argument(
        "-cp",
        "--checkpoint",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint filename (default: checkpoint_final.pth)",
    )
    parser.add_argument("-o", "--output_path", type=str, help="Custom output .onnx path")
    parser.add_argument("-d_device", "--device", type=str, help="Torch device, e.g. cuda:0")
    parser.add_argument(
        "-no_da",
        "--no_dynamic_axes",
        action="store_true",
        help="Disable dynamic batch+spatial axes (export fixed-shape ONNX)",
    )
    parser.add_argument(
        "-is",
        "--input_shape",
        type=int,
        nargs="+",
        help="Custom input shape, e.g. 1 1 128 128 128",
    )
    parser.add_argument("-sim", "--simplify", action="store_true", help="Run onnx-simplifier")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "-da5", "--use_da5", action="store_true", help="Set if checkpoint was trained with DA5"
    )

    args = parser.parse_args()

    export_primus_distillation_to_onnx(
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        teacher_size=args.teacher_size,
        feature_reduction_factor=args.reduction_factor,
        checkpoint_name=args.checkpoint,
        output_path=args.output_path,
        device=args.device,
        dynamic_axes=not args.no_dynamic_axes,
        input_shape=tuple(args.input_shape) if args.input_shape else None,
        simplify_onnx=args.simplify,
        verbose=args.verbose,
        use_da5=args.use_da5,
    )


if __name__ == "__main__":
    main()
