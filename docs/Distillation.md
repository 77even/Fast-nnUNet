<div align="center">
  <img src="../docs/assets/fastnnunet-logo.png" alt="FastnnUNet Logo" width="600">
</div>

# 🧪 FastnnUNet Knowledge Distillation Training Module

The FastnnUNet Knowledge Distillation Module is an advanced knowledge transfer system developed based on the nnUNetv2 framework. It transfers knowledge from high-performance but computationally complex teacher models (standard nnUNet) to lightweight student models, achieving significant reductions in model size and computational cost while maintaining segmentation accuracy.

## ✨ Core Features

- **Efficient Knowledge Distillation**: Transfers segmentation capabilities from standard nnUNet to lightweight models
- **Multi-teacher Ensemble Learning**: Supports knowledge extraction from multiple cross-validation model ensembles
- **Adaptive Network Architecture**: Automatically designs optimized student networks based on teacher models
- **Hybrid Distillation Strategy**: Joint training method combining soft labels and hard labels
- **Feature Reduction Control**: Configurable feature channel reduction ratio (default 50% reduction)
- **DA5 Strong Data Augmentation**: Advanced data augmentation strategy specifically optimized for small datasets
- **Compatibility Guarantee**: Fully compatible with original nnUNetv2, supporting all configurations and datasets
- **Complete Training Cycle**: Includes checkpoint recovery, automatic validation, and ONNX export functionality
- **ResEnc Architecture Support**: Enhanced support for Residual Encoder U-Net (ResEnc) architectures with improved performance for complex segmentation tasks

## Technical Principles

FastnnUNet knowledge distillation employs advanced knowledge transfer strategies, mainly including:

1. **Soft Label Knowledge Transfer**:
   - Extract class probability distributions rather than hard labels from teacher models
   - Control the "softness" of soft labels using temperature parameters
   - Capture knowledge of correlations between categories through KL divergence loss

2. **Architecture Optimization Design**:
   - Maintain the U-Net topology structure of the teacher model
   - Proportionally reduce the number of feature channels in each layer
   - Retain key design elements of the teacher model (depth, convolution types, skip connections, etc.)

3. **Multi-teacher Ensemble Learning**:
   - Simultaneously leverage ensemble knowledge from probabilistic maps of multiple teacher models (different folds), instead of having to reason about each of the 5 folded models once and pool the predictions
   - Simplify and get rid of nnUNet's original idea of finding the best model after training a 5-fold model
   - Capture complementary information between models, improving generalization ability

4. **ResEnc Architecture Support**:
   - Support for Residual Encoder U-Net architectures with deeper encoder structures
   - Enhanced feature extraction capabilities through residual connections in the encoder
   - Multiple ResEnc variants: ResEncM, ResEncL, and ResEncXL for different complexity requirements
   - Improved performance for complex medical image segmentation tasks with challenging anatomical structures

5. **DA5 Strong Data Augmentation**:
   - Advanced data augmentation strategy optimized for small datasets and challenging segmentation tasks
   - Enhanced spatial transformations with reduced interpolation order for better preservation of fine details
   - Comprehensive augmentation pipeline including rotation, scaling, elastic deformation, and intensity transformations
   - Particularly effective for medical imaging datasets with limited training samples
   - Compatible with both standard and ResEnc distillation architectures

## 🔧 Installation Requirements

Before using the FastnnUNet Knowledge Distillation Module, please ensure the following requirements are met:

1. **Basic Environment**:
   - Python 3.7+
   - CUDA 11.0+ (recommended for GPU training)
   - PyTorch 1.11+

2. **nnUNetv2 Dependencies**:
   - Ensure nnUNetv2 is correctly installed
   - Environment variables are properly configured

3. **Install This Module**:
   ```bash
   # Enter the project directory
   cd FastnnUNet/distillation
   
   # Install in development mode
   pip install -e .
   ```

## 📋 How to use?

### 1. Data Preparation and Teacher Model Training

First, prepare the dataset and train standard nnUNetv2 models as teachers:

```bash
# Data preprocessing
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train 5-fold cross-validation teacher models (Standard nnUNet)
nnUNetv2_train DATASET_ID 3d_fullres 0
nnUNetv2_train DATASET_ID 3d_fullres 1
nnUNetv2_train DATASET_ID 3d_fullres 2
nnUNetv2_train DATASET_ID 3d_fullres 3
nnUNetv2_train DATASET_ID 3d_fullres 4

# Alternative: Train ResEnc teacher models for enhanced performance
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -pl nnUNetPlannerResEncM/nnUNetPlannerResEncL/nnUNetPlannerResEncXL

# If you only intend to run 3d_fullres and 2d and have already preprocessed these datasets, to avoid reprocessing, run directly: 
nnUNetv2_plan_experiment -d DATASET_ID -pl nnUNetPlannerResEncM/nnUNetPlannerResEncL/nnUNetPlannerResEncXL

nnUNetv2_train DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 1 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 2 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 3 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 4 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans

```

### 2. Knowledge Distillation Training

Use the trained teacher models for knowledge distillation.

#### Supported nnUNet Configurations

All four standard nnUNetv2 configurations are supported via `-c / --configuration`:

| Configuration | Description | Notes |
| --- | --- | --- |
| `2d` | 2D slice-based U-Net | Uses Conv2d / InstanceNorm2d automatically |
| `3d_lowres` | Low-resolution 3D U-Net | Independent stage, also used as cascade stage 1 |
| `3d_fullres` | Full-resolution 3D U-Net (default) | Most common choice |
| `3d_cascade_fullres` | High-res second stage of the cascade | Reads `predicted_next_stage/` from the corresponding `3d_lowres` model |

The student network's spatial dimensionality (2D vs 3D) and the cascade input-channel count are detected automatically from the configuration's `patch_size` and the upstream plans — no extra flags are needed.

**Cascade workflow:** train and predict the lowres stage with stock nnUNetv2 first so the cascade inputs exist, then run distillation on each stage:

```bash
# (1) Train and predict the upstream lowres teacher with stock nnUNetv2,
#     which populates predicted_next_stage/3d_cascade_fullres/.
nnUNetv2_train DATASET_ID 3d_lowres 0
nnUNetv2_predict_from_modelfolder ...  # see nnUNetv2 docs for cascade prep

# (2) Distill the lowres stage (optional; you can also keep the upstream lowres model).
nnUNetv2_distillation_train -d DATASET_ID -c 3d_lowres -f 0 -a 0.3 -temp 3.0 -r 2

# (3) Distill the cascade fullres stage — extra input channels are picked up automatically.
nnUNetv2_distillation_train -d DATASET_ID -c 3d_cascade_fullres -f 0 -a 0.3 -temp 3.0 -r 2
```

#### Standard Knowledge Distillation

```bash
# Basic usage (using all available folds of teacher models)
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Advanced usage - specify specific teacher model folds
nnUNetv2_distillation_train -d DATASET_ID -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2

# Continue previous training
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -c_continue

# With custom teacher model path
nnUNetv2_distillation_train -d DATASET_ID -f 0 -t /path/to/teacher -a 0.3 -temp 3.0 -r 2

# Disable mirroring during validation
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -disable_mirroring

# Enable fold rotation during training
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -rotate_folds -rotate_freq 400

# Use DA5 strong data augmentation (recommended for small datasets)
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 --use_da5

# Combine DA5 with other options
nnUNetv2_distillation_train -d DATASET_ID -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2 --use_da5 -c_continue

# 2D / 3d_lowres / 3d_cascade_fullres examples (default is 3d_fullres)
nnUNetv2_distillation_train -d DATASET_ID -c 2d                 -f 0 -a 0.3 -temp 3.0 -r 2
nnUNetv2_distillation_train -d DATASET_ID -c 3d_lowres          -f 0 -a 0.3 -temp 3.0 -r 2
nnUNetv2_distillation_train -d DATASET_ID -c 3d_cascade_fullres -f 0 -a 0.3 -temp 3.0 -r 2
```

#### ResEnc Knowledge Distillation (Enhanced Performance)

```bash
# Basic ResEnc distillation (using ResEnc teacher models)
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Advanced ResEnc distillation - specify teacher plans and folds
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetLPlans

# ResEnc distillation with different teacher plan variants
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetMPlans  # Medium
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetLPlans  # Large
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetXLPlans # Extra Large

# ResEnc distillation with custom teacher model path
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -t /path/to/resenc/teacher -a 0.3 -temp 3.0 -r 2

# ResEnc distillation with different student plans (ResEnc student)
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -spl nnUNetResEncUNetMPlans

# ResEnc distillation with block reduction strategy
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -bs adaptive

# Continue ResEnc distillation training
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -c_continue

# ResEnc distillation with DA5 strong data augmentation
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 --use_da5

# ResEnc distillation combining DA5 with ResEnc teacher and student
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetLPlans -spl nnUNetResEncUNetMPlans --use_da5
```

#### Primus Knowledge Distillation

Distill an upstream nnUNetv2 Primus transformer teacher into a smaller Primus student. Primus
is 3D-only and requires `patch_size` divisible by 8.

Four teacher sizes are supported (select with `-ts / --teacher_size`):

| Size | embed_dim | depth | num_heads | head_dim |
| --- | --- | --- | --- | --- |
| `S` | 396 | 12 | 6 | 66 |
| `B` | 792 | 12 | 12 | 66 |
| `M` (default) | 864 | 16 | 12 | 72 |
| `L` | 1056 | 24 | 16 | 66 |

The student is produced by holding `head_dim` constant and dividing `num_heads` / `depth` by
`-r / --reduction_factor`. For example, `-ts M -r 2` yields a student with embed_dim=432,
depth=8, num_heads=6 (head_dim still 72, so the 3D rotary positional embedding stays valid).

Teacher folder is auto-derived from `-ts`:
```
{nnUNet_results}/{Dataset}/nnUNet_Primus_{S|B|M|L}_Trainer__nnUNetPlans__{configuration}/
```

```bash
# 1) Train the upstream Primus teacher with stock nnUNetv2 first:
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNet_Primus_M_Trainer
# (repeat for folds 1-4 if you want a multi-teacher ensemble)

# 2) Distill into a Primus student (default teacher size M, reduction 2x):
nnUNetv2_primus_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Pick a smaller / larger teacher:
nnUNetv2_primus_distillation_train -d DATASET_ID -ts S -f 0 -a 0.3 -temp 3.0 -r 2
nnUNetv2_primus_distillation_train -d DATASET_ID -ts L -f 0 -a 0.3 -temp 3.0 -r 2

# Multi-teacher ensemble across folds:
nnUNetv2_primus_distillation_train -d DATASET_ID -ts M -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2

# Continue previous training, custom warmup duration, DA5 augmentation:
nnUNetv2_primus_distillation_train -d DATASET_ID -ts M -f 0 -r 2 -w 50 -c_continue
nnUNetv2_primus_distillation_train -d DATASET_ID -ts M -f 0 -r 2 --use_da5

# Custom teacher folder (e.g. one of the BS8 / 96_BS1 variants):
nnUNetv2_primus_distillation_train -d DATASET_ID -ts M -f 0 -r 2 -t /path/to/custom/primus/teacher
```

Primus-specific notes:
- The student uses AdamW + linear warmup → polynomial schedule (matches upstream
  `AbstractPrimus`); `-w / --warmup_epochs` controls the warmup duration (default 50).
- Deep supervision is disabled — Primus emits a single full-resolution map.
- Cascade configurations (`3d_cascade_fullres`) are accepted as long as the underlying
  Primus teacher was trained on the same configuration with the lowres predictions in place.

Parameter description:
- `-d, --dataset_id`: Dataset ID
- `-f, --fold`: Fold number used to train the student model
- `-tf, --teacher_folds`: List of teacher model fold numbers, automatically detected if not specified
- `-a, --alpha`: Distillation loss weight, controls the ratio of soft label and hard label losses
- `-temp, --temperature`: Distillation temperature, controls the smoothness of soft labels
- `-r, --reduction_factor`: Feature reduction factor, higher values produce smaller student models
- `-c_continue, --continue_training`: Continue previous training
- `-disable_mirroring`: Disable mirror augmentation during validation
- `-e, --epochs`: Maximum number of training epochs (default 1000)
- `-t, --teacher_model_folder`: Teacher model folder path (auto-constructed if not provided)
- `-tpl, --teacher_plans`: Teacher model plans identifier (ResEnc specific, default: nnUNetResEncUNetLPlans)
- `-spl, --student_plans`: Student model plans identifier (ResEnc specific, default: nnUNetPlans)
- `-bs, --block_strategy`: Block reduction strategy for ResEnc (reduce/keep/increase/adaptive, default: keep)
- `-rotate_folds`: Enable rotating training folds periodically
- `-rotate_freq`: How often to rotate folds (in epochs, default: 5 for ResEnc, 400 for standard)
- `-d_device`: Device to use, e.g., "cuda:0"
- `--use_da5`: Use DA5 strong data augmentation (recommended for small datasets)

### 3. 📤 Export ONNX Model

Export the trained student model to ONNX format for fast inference:

#### Standard Distillation Model Export

```bash
# Basic export
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2

# Verbose information display
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -v

# Specify output path
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -o /path/to/output.onnx

# Export with daynmic input shape
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -da

# Export with custom input shape
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -is 1 1 128 128 128

# Export in nnUNet format (single channel, fixed size)
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -nn

# Export DA5-trained model
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -da5

# Export DA5-trained model with verbose output
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -da5 -v

# Export with simplified ONNX
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -sim

# Export 2D / 3d_lowres / 3d_cascade_fullres models — 2D dumps an N×C×H×W ONNX,
# 3d_cascade_fullres uses the lowres-augmented input channel count automatically.
nnUNetv2_distillation_export_onnx -d DATASET_ID -c 2d                 -f 0 -r 2
nnUNetv2_distillation_export_onnx -d DATASET_ID -c 3d_lowres          -f 0 -r 2
nnUNetv2_distillation_export_onnx -d DATASET_ID -c 3d_cascade_fullres -f 0 -r 2
```

#### ResEnc Distillation Model Export

```bash
# Basic ResEnc model export
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2

# ResEnc export with verbose output
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -v

# ResEnc export with custom output directory
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -o /path/to/output/dir

# ResEnc export with specific student plans
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -spl nnUNetResEncUNetMPlans

# ResEnc export with fixed batch size
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -b 1

# ResEnc export with TensorRT compatibility
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -fix

# ResEnc export with custom plans identifier
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -p nnUNetResEncUNetLPlans

# ResEnc export for DA5-trained model
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -da5

# ResEnc export for DA5-trained model with TensorRT compatibility
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -da5 -fix

# ResEnc export with simplified ONNX
nnUNetv2_resenc_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -sim
```

#### Primus Distillation Model Export

```bash
# Basic export (default teacher size M, reduction 2x)
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -f 0 -r 2

# Pick a different teacher size used during training
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts S -f 0 -r 2
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts L -f 0 -r 2

# Custom output path
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts M -f 0 -r 2 -o /path/to/primus.onnx

# Custom input shape (must satisfy patch_size divisibility constraints of the teacher)
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts M -f 0 -r 2 -is 1 1 128 128 128

# Fixed-shape export (disable dynamic axes)
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts M -f 0 -r 2 -no_da

# Export DA5-trained checkpoint
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts M -f 0 -r 2 -da5

# Simplify with onnx-simplifier
nnUNetv2_primus_distillation_export_onnx -d DATASET_ID -ts M -f 0 -r 2 -sim
```

Parameter description:
- `-d, --dataset_id`: Dataset ID
- `-f, --fold`: Model fold number
- `-r, --reduction_factor`: Feature reduction factor, must be consistent with training
- `-o, --output`: Output ONNX file path (standard) or output directory (ResEnc)
- `-v, --verbose`: Display detailed information
- `-da`: Use dynamic input shape (default uses static shape)
- `-is`: Custom input shape (b c x y z) - standard export only
- `-nn`: Export single channel fixed size model - standard export only
- `-spl, --student_plans`: Student model plans identifier - ResEnc export only
- `-b, --batch_size`: Batch size, 0 means dynamic - ResEnc export only
- `--trt`: Apply TensorRT compatibility fixes - ResEnc export only
- `-p, --plans`: Plans identifier - ResEnc export only
- `-da5`: Model was trained with DA5 data augmentation (both standard and ResEnc export)
- `-sim`: Simplify ONNX model

## Parameter Tuning Recommendations

For the best balance between performance and accuracy, the following parameter settings are recommended:

1. 📊 **Feature Reduction Factor (`-r`)**:
   - 2: Balanced mode, model size reduced by 75%, minimal accuracy loss
   - 4: Lightweight mode, model size reduced by 94%, may have 1-2% accuracy loss
   - 8: Ultra-lightweight mode, model size reduced by 98%, suitable for extremely resource-constrained scenarios

2. ⚖️ **Distillation Loss Weight (`-a`)**:
   - 0.3: Default value, suitable for most scenarios
   - 0.5: More emphasis on soft label knowledge, suitable for complex segmentation tasks
   - 0.1: More emphasis on hard labels, suitable for simple segmentation tasks

3. 🌡️ **Distillation Temperature (`-temp`)**:
   - 3.0: Default value, provides moderately smooth soft labels
   - 1.0: Sharper soft labels, closer to original predictions
   - 5.0: Very smooth soft labels, maximizes knowledge transfer

4. 🔥 **DA5 Data Augmentation (`--use_da5`)**:
   - Recommended for small datasets (< 100 training cases)
   - Particularly effective for challenging segmentation tasks with fine anatomical structures
   - Can improve model robustness by 2-5% on small datasets
   - Compatible with all reduction factors and architectures
   - May increase training time by 10-15% due to more intensive augmentation


## 🚀 Advanced Usage

### Using Custom Teacher Models

If you have teacher models trained elsewhere, you can specify them using the `-t` parameter:

```bash
# Standard distillation with custom teacher
nnUNetv2_distillation_train -d DATASET_ID -f 0 -t /path/to/teacher/model -a 0.3 -temp 3.0 -r 2

# ResEnc distillation with custom teacher
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -t /path/to/resenc/teacher/model -a 0.3 -temp 3.0 -r 2
```

### ResEnc Architecture Variants

Choose different ResEnc architectures based on your computational requirements:

```bash
# ResEnc Medium (balanced performance and efficiency)
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -tpl nnUNetResEncUNetMPlans -a 0.3 -temp 3.0 -r 2

# ResEnc Large (higher performance for complex tasks)
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -tpl nnUNetResEncUNetLPlans -a 0.3 -temp 3.0 -r 2

# ResEnc Extra Large (maximum performance)
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -tpl nnUNetResEncUNetXLPlans -a 0.3 -temp 3.0 -r 2
```

### 💪 Multi-GPU Training

Although student models are usually smaller, multi-GPU training is still supported:

```bash
# Standard distillation multi-GPU
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# ResEnc distillation multi-GPU
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Specify specific GPU device
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -d_device cuda:0
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -d_device cuda:1

# Multi-GPU training with DA5 for small datasets
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 --use_da5
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 --use_da5
```

### 🔥 DA5 Strong Data Augmentation

DA5 is an advanced data augmentation strategy specifically designed for small datasets and challenging segmentation tasks. It provides stronger augmentation compared to the default nnUNet augmentation.

#### When to Use DA5

- **Small datasets**: Datasets with fewer than 100 training cases
- **Challenging segmentation tasks**: Tasks with fine anatomical structures or difficult-to-segment regions
- **Limited training data**: When you want to maximize the use of available training samples
- **Improved robustness**: When you need models that generalize better to unseen data variations

#### DA5 Features

- **Enhanced spatial transformations**: More aggressive rotation, scaling, and elastic deformation
- **Reduced interpolation order**: Better preservation of fine details during augmentation
- **Comprehensive intensity augmentation**: Advanced brightness, contrast, and gamma transformations
- **Optimized for medical imaging**: Specifically tuned for medical image characteristics

#### Example Usage Scenarios

```bash
# Small dataset scenario (e.g., rare disease with <50 cases)
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.5 -temp 4.0 -r 2 --use_da5

# Challenging anatomy (e.g., small organ segmentation)
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.4 -temp 3.5 -r 2 -tpl nnUNetResEncUNetLPlans --use_da5

# Combining DA5 with other advanced features
nnUNetv2_resenc_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 --use_da5 -rotate_folds -bs adaptive
```

## 🔧 Troubleshooting

1. **Out of Memory Errors**:
   - Reduce batch size: The trainer will automatically select an appropriate batch size
   - Use a larger feature reduction factor: Try `-r 4` or higher

2. **Excessive Accuracy Decline**:
   - Reduce the feature reduction factor: Try `-r 2` or lower
   - Increase training epochs: Use `-e 1500` or more
   - Adjust temperature parameter: Try `-temp 2.0`

3. **Training Instability**:
   - Adjust distillation weight: Try `-a 0.4` or `-a 0.2`
   - Ensure teacher model quality: Check teacher model performance

4. **ResEnc Specific Issues**:
   - **Teacher model not found**: Ensure ResEnc teacher models are properly trained with correct plans identifier
   - **Architecture mismatch**: Verify teacher and student plans compatibility when using ResEnc variants
   - **Memory issues with large ResEnc models**: Consider using ResEncM instead of ResEncL/ResEncXL for limited GPU memory

5. **DA5 Related Issues**:
   - **Slower training with DA5**: This is expected due to more intensive augmentation; consider reducing batch size if needed
   - **DA5 not improving results**: DA5 is most effective on small datasets; may not provide benefits on large datasets (>500 cases)
   - **Export issues with DA5 models**: Ensure you use `--use_da5` flag when exporting models trained with DA5
   - **Memory issues with DA5**: The stronger augmentation may require more GPU memory; reduce batch size or use gradient accumulation

## 📝 Citation

If you use FastnnUNet Knowledge Distillation in your research, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
``` 
