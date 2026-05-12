#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0

## Title: Fast-nnUNet — Primus knowledge-distillation trainer
## Authors: Justin Lee
## Description: Distillation trainer + lightweight student for the upstream nnunetv2 Primus
##              transformer architecture (S/B/M/L teachers).

"""Primus-flavored knowledge-distillation trainer.

Provides:
- ``reduce_primus_dims``: shrink (embed_dim, depth, num_heads) by an integer factor while
  preserving the ``embed_dim % num_heads == 0`` constraint required by Primus' attention.
- ``LitePrimusStudent``: thin wrapper around
  ``dynamic_network_architectures.architectures.primus.Primus`` that accepts the reduced
  hyperparameters.
- ``nnUNetDistillationPrimusTrainer`` / ``nnUNetDistillationPrimusTrainerDA5``: distillation
  trainers that reuse the existing multi-teacher KL pipeline from
  ``nnUNetDistillationTrainer`` but build a Primus student and use upstream Primus' AdamW +
  warmup optimizer schedule.

The teacher must be a trained upstream Primus model
(``nnUNet_Primus_{S,B,M,L}_Trainer__nnUNetPlans__<configuration>``). Multi-teacher ensembling,
fold rotation, and DA5 augmentation are inherited unchanged from the existing distillation
trainer family.
"""

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure the vendored nnunetv2 tree (on main) wins over a pip-installed copy.
# On the refactor branch this is a no-op because the module lives next to the entry-point
# scripts and the vendored tree is gone, but keeping the insertion is harmless.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_network_architectures.architectures.primus import Primus
from nnunetv2.training.lr_scheduler.warmup import (
    Lin_incr_LRScheduler,
    PolyLRScheduler_offset,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
)
from nnunetv2.training.nnUNetTrainer.variants.nnUNetDistillationTrainer import (
    nnUNetDistillationTrainer,
)
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


# Reference architecture specs for each upstream Primus size.
PRIMUS_TEACHER_SPECS = {
    "S": dict(embed_dim=396, depth=12, num_heads=6),
    "B": dict(embed_dim=792, depth=12, num_heads=12),
    "M": dict(embed_dim=864, depth=16, num_heads=12),
    "L": dict(embed_dim=1056, depth=24, num_heads=16),
}

# Upstream Primus trainer class names, used to construct the default teacher folder.
PRIMUS_TEACHER_TRAINER_CLASS = {
    "S": "nnUNet_Primus_S_Trainer",
    "B": "nnUNet_Primus_B_Trainer",
    "M": "nnUNet_Primus_M_Trainer",
    "L": "nnUNet_Primus_L_Trainer",
}


def reduce_primus_dims(
    embed_dim: int, depth: int, num_heads: int, factor: int
) -> Tuple[int, int, int]:
    """Shrink Primus hyperparameters by ``factor`` while keeping the model valid.

    Primus' 3D rotary positional embedding requires ``head_dim`` (= embed_dim // num_heads)
    to be divisible by 6. The reliable way to keep that invariant under reduction is to
    **hold ``head_dim`` constant at the teacher's value** and shrink only ``num_heads``
    (and ``depth``):

    - ``num_heads`` is floor-divided by ``factor`` but never goes below 1.
    - ``embed_dim`` is set to ``head_dim * num_heads`` so head_dim is preserved exactly.
    - ``depth`` is floor-divided by ``factor`` but never goes below 2 — we still want at
      least one transformer block with skip + norm.
    """
    if factor < 1:
        raise ValueError(f"reduction factor must be >= 1, got {factor}")
    if num_heads < 1 or embed_dim % num_heads != 0:
        raise ValueError(
            f"teacher embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
    head_dim = embed_dim // num_heads
    new_heads = max(num_heads // factor, 1)
    new_embed = head_dim * new_heads
    new_depth = max(depth // factor, 2)
    return new_embed, new_depth, new_heads


class LitePrimusStudent(nn.Module):
    """Primus with reduced embed_dim/depth/heads for knowledge distillation."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: Tuple[int, ...],
        patch_embed_size: Tuple[int, ...] = (8, 8, 8),
        drop_path_rate: float = 0.2,
        init_values: float = 0.1,
        scale_attn_inner: bool = True,
    ):
        super().__init__()
        if any(p % 8 != 0 for p in patch_size):
            raise ValueError(
                f"Primus requires patch_size divisible by 8; got {tuple(patch_size)}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = tuple(patch_size)
        self.patch_embed_size = tuple(patch_embed_size)
        self.network = Primus(
            input_channels,
            embed_dim,
            tuple(patch_embed_size),
            num_classes,
            eva_depth=depth,
            eva_numheads=num_heads,
            input_shape=tuple(patch_size),
            drop_path_rate=drop_path_rate,
            scale_attn_inner=scale_attn_inner,
            init_values=init_values,
        )

    def forward(self, x):
        return self.network(x)


class nnUNetDistillationPrimusTrainer(nnUNetDistillationTrainer):
    """Knowledge-distillation trainer with a Primus student.

    Inherits the multi-teacher / KL-loss / fold-rotation machinery from
    ``nnUNetDistillationTrainer`` and swaps in:
    - A Primus student instead of a CNN U-Net student.
    - AdamW + warmup optimizer schedule (matches upstream ``AbstractPrimus``).
    - Deep supervision disabled — Primus emits a single full-resolution map.
    """

    DEFAULT_WARMUP_DURATION_WHOLE_NET = 50

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        teacher_model_folder=None,
        teacher_fold=0,
        teacher_checkpoint_name="checkpoint_final.pth",
        alpha=0.3,
        temperature=3.0,
        feature_reduction_factor=2,
        rotate_training_folds=False,
        rotate_folds_frequency=5,
        teacher_size="M",
        warmup_duration_whole_net=DEFAULT_WARMUP_DURATION_WHOLE_NET,
        device=torch.device("cuda"),
    ):
        if teacher_size not in PRIMUS_TEACHER_SPECS:
            raise ValueError(
                f"teacher_size must be one of {sorted(PRIMUS_TEACHER_SPECS)}; got {teacher_size!r}"
            )
        # block_reduction_strategy is ignored for Primus (no residual blocks); we pass a
        # placeholder so the parent constructor doesn't complain.
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            teacher_model_folder=teacher_model_folder,
            teacher_fold=teacher_fold,
            teacher_checkpoint_name=teacher_checkpoint_name,
            alpha=alpha,
            temperature=temperature,
            feature_reduction_factor=feature_reduction_factor,
            block_reduction_strategy="keep",
            rotate_training_folds=rotate_training_folds,
            rotate_folds_frequency=rotate_folds_frequency,
            device=device,
        )

        # Primus-specific overrides (parent's patched_init flipped some of these on).
        self.enable_deep_supervision = False
        self.teacher_size = teacher_size
        self.warmup_duration_whole_net = warmup_duration_whole_net
        self.training_stage = None
        # Mirror upstream AbstractPrimus defaults.
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.my_init_kwargs.update(
            {
                "teacher_size": teacher_size,
                "warmup_duration_whole_net": warmup_duration_whole_net,
            }
        )

    # ------------------------------------------------------------------
    # Network architecture
    # ------------------------------------------------------------------
    def build_network_architecture(
        self,
        architecture_class_name=None,
        arch_init_kwargs=None,
        arch_init_kwargs_req_import=None,
        num_input_channels: int = None,
        num_output_channels: int = None,
        enable_deep_supervision: bool = False,
    ):
        if num_input_channels is None:
            num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
        if num_output_channels is None:
            num_output_channels = self.label_manager.num_segmentation_heads

        patch_size = tuple(self.configuration_manager.patch_size)
        if len(patch_size) != 3:
            raise ValueError(
                f"Primus is 3D-only; got patch_size {patch_size} (configuration={self.configuration_name})"
            )
        if any(p % 8 != 0 for p in patch_size):
            raise ValueError(
                f"Primus requires patch_size divisible by 8; got {patch_size}. "
                "Pick a configuration whose patch_size matches, or adjust the plans."
            )

        teacher_spec = PRIMUS_TEACHER_SPECS[self.teacher_size]
        s_embed, s_depth, s_heads = reduce_primus_dims(
            teacher_spec["embed_dim"],
            teacher_spec["depth"],
            teacher_spec["num_heads"],
            self.feature_reduction_factor,
        )
        self.print_to_log_file(
            f"Primus student: embed_dim={s_embed}, depth={s_depth}, num_heads={s_heads} "
            f"(reduced from teacher {self.teacher_size} "
            f"embed_dim={teacher_spec['embed_dim']}, depth={teacher_spec['depth']}, "
            f"num_heads={teacher_spec['num_heads']}, factor={self.feature_reduction_factor})"
        )
        return LitePrimusStudent(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            embed_dim=s_embed,
            depth=s_depth,
            num_heads=s_heads,
            patch_size=patch_size,
        )

    # ------------------------------------------------------------------
    # Optimizer / scheduler — mirrors upstream AbstractPrimus
    # ------------------------------------------------------------------
    def configure_optimizers(self, stage: str = "warmup_all"):
        if stage not in {"warmup_all", "train"}:
            raise ValueError(f"unknown stage {stage!r}")
        if self.training_stage == stage and self.optimizer is not None:
            return self.optimizer, self.lr_scheduler

        params = (
            self.network.module.parameters()
            if isinstance(self.network, DDP)
            else self.network.parameters()
        )
        if stage == "warmup_all":
            optimizer = torch.optim.AdamW(
                params,
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
            )
            lr_scheduler = Lin_incr_LRScheduler(
                optimizer, self.initial_lr, self.warmup_duration_whole_net
            )
            self.print_to_log_file(
                f"Primus distill: warmup_all optimizer at epoch {self.current_epoch}"
            )
        else:
            optimizer = torch.optim.AdamW(
                params,
                self.initial_lr,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
            )
            lr_scheduler = PolyLRScheduler_offset(
                optimizer,
                self.initial_lr,
                self.num_epochs,
                self.warmup_duration_whole_net,
            )
            self.print_to_log_file(
                f"Primus distill: train optimizer at epoch {self.current_epoch}"
            )
        self.training_stage = stage
        return optimizer, lr_scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        # Primus is single-resolution; this is a no-op.
        return


class nnUNetDistillationPrimusTrainerDA5(
    nnUNetDistillationPrimusTrainer, nnUNetTrainerDA5
):
    """Primus distillation trainer with DA5 strong data augmentation."""

    def __init__(self, *args, **kwargs):
        nnUNetDistillationPrimusTrainer.__init__(self, *args, **kwargs)
        self.print_to_log_file(
            "Using DA5 strong data augmentation for Primus knowledge distillation"
        )
