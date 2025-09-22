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
nnUNetv2_train DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 1 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 2 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 3 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
nnUNetv2_train DATASET_ID 3d_fullres 4 -p nnUNetResEncUNetMPlans/nnUNetResEncUNetLPlans/nnUNetResEncUNetXLPlans
```

### 2. Knowledge Distillation Training

Use the trained teacher models for knowledge distillation:

#### Standard Knowledge Distillation

```bash
# Basic usage (using all available folds of teacher models)
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Advanced usage - specify specific teacher model folds
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2

# Continue previous training
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -c_continue

# With custom teacher model path
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -t /path/to/teacher -a 0.3 -temp 3.0 -r 2

# Disable mirroring during validation
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -disable_mirroring

# Enable fold rotation during training
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -rotate_folds -rotate_freq 400
```

#### ResEnc Knowledge Distillation (Enhanced Performance)

```bash
# Basic ResEnc distillation (using ResEnc teacher models)
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Advanced ResEnc distillation - specify teacher plans and folds
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetLPlans

# ResEnc distillation with different teacher plan variants
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetMPlans  # Medium
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetLPlans  # Large
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -tpl nnUNetResEncUNetXLPlans # Extra Large

# ResEnc distillation with custom teacher model path
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -t /path/to/resenc/teacher -a 0.3 -temp 3.0 -r 2

# ResEnc distillation with different student plans (ResEnc student)
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -spl nnUNetResEncUNetMPlans

# ResEnc distillation with block reduction strategy
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -bs adaptive

# Continue ResEnc distillation training
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -c_continue
```

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

### 3. 📤 Export ONNX Model

Export the trained student model to ONNX format for fast inference:

#### Standard Distillation Model Export

```bash
# Basic export
python distillation/fast_nnunet_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2

# Verbose information display
python distillation/fast_nnunet_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -v

# Specify output path
python distillation/fast_nnunet_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -o /path/to/output.onnx

# Export with static input shape
python distillation/fast_nnunet_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 --static

# Export with custom input shape
python distillation/fast_nnunet_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 --input_shape 1 1 128 128 128

# Export in nnUNet format (single channel, fixed size)
python distillation/fast_nnunet_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 --nnunet_format
```

#### ResEnc Distillation Model Export

```bash
# Basic ResEnc model export
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2

# ResEnc export with verbose output
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -v

# ResEnc export with custom output directory
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -o /path/to/output/dir

# ResEnc export with specific student plans
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -spl nnUNetResEncUNetMPlans

# ResEnc export with fixed batch size
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -b 1

# ResEnc export with TensorRT compatibility
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 --trt

# ResEnc export with custom plans identifier
python distillation/fast_nnunet_resenc_distillation_export_onnx.py -d DATASET_ID -f 0 -r 2 -p nnUNetResEncUNetLPlans
```

Parameter description:
- `-d, --dataset_id`: Dataset ID
- `-f, --fold`: Model fold number
- `-r, --reduction_factor`: Feature reduction factor, must be consistent with training
- `-o, --output`: Output ONNX file path (standard) or output directory (ResEnc)
- `-v, --verbose`: Display detailed information
- `--static`: Use static input shape (default uses dynamic shape)
- `--input_shape`: Custom input shape (b c x y z) - standard export only
- `--nnunet_format`: Export single channel fixed size model - standard export only
- `-spl, --student_plans`: Student model plans identifier - ResEnc export only
- `-b, --batch_size`: Batch size, 0 means dynamic - ResEnc export only
- `--trt`: Apply TensorRT compatibility fixes - ResEnc export only
- `-p, --plans`: Plans identifier - ResEnc export only

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


## 🚀 Advanced Usage

### Using Custom Teacher Models

If you have teacher models trained elsewhere, you can specify them using the `-t` parameter:

```bash
# Standard distillation with custom teacher
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -t /path/to/teacher/model -a 0.3 -temp 3.0 -r 2

# ResEnc distillation with custom teacher
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -t /path/to/resenc/teacher/model -a 0.3 -temp 3.0 -r 2
```

### ResEnc Architecture Variants

Choose different ResEnc architectures based on your computational requirements:

```bash
# ResEnc Medium (balanced performance and efficiency)
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -tpl nnUNetResEncUNetMPlans -a 0.3 -temp 3.0 -r 2

# ResEnc Large (higher performance for complex tasks)
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -tpl nnUNetResEncUNetLPlans -a 0.3 -temp 3.0 -r 2

# ResEnc Extra Large (maximum performance)
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -tpl nnUNetResEncUNetXLPlans -a 0.3 -temp 3.0 -r 2
```

### 💪 Multi-GPU Training

Although student models are usually smaller, multi-GPU training is still supported:

```bash
# Standard distillation multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# ResEnc distillation multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Specify specific GPU device
python distillation/fast_nnunet_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -d_device cuda:0
python distillation/fast_nnunet_resenc_distillation_train.py -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -d_device cuda:1
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

## 📝 Citation

If you use FastnnUNet Knowledge Distillation in your research, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
``` 
