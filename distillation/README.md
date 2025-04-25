# FastnnUNet Knowledge Distillation Training Module

The FastnnUNet Knowledge Distillation Module is an advanced knowledge transfer system developed based on the nnUNetv2 framework. It transfers knowledge from high-performance but computationally complex teacher models (standard nnUNet) to lightweight student models, achieving significant reductions in model size and computational cost while maintaining segmentation accuracy.

## Core Features

- **Efficient Knowledge Distillation**: Transfers segmentation capabilities from standard nnUNet to lightweight models
- **Multi-teacher Ensemble Learning**: Supports knowledge extraction from multiple cross-validation model ensembles
- **Adaptive Network Architecture**: Automatically designs optimized student networks based on teacher models
- **Hybrid Distillation Strategy**: Joint training method combining soft labels and hard labels
- **Feature Reduction Control**: Configurable feature channel reduction ratio (default 50% reduction)
- **Compatibility Guarantee**: Fully compatible with original nnUNetv2, supporting all configurations and datasets
- **Complete Training Cycle**: Includes checkpoint recovery, automatic validation, and ONNX export functionality

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
   - Simultaneously leverage ensemble knowledge from multiple teacher models (different folds)
   - Capture complementary information between models, improving generalization ability

## Installation Requirements

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

## Detailed Usage Workflow

### 1. Data Preparation and Teacher Model Training

First, prepare the dataset and train standard nnUNetv2 models as teachers:

```bash
# Data preprocessing
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train 5-fold cross-validation teacher models
nnUNetv2_train DATASET_ID 3d_fullres 0
nnUNetv2_train DATASET_ID 3d_fullres 1
nnUNetv2_train DATASET_ID 3d_fullres 2
nnUNetv2_train DATASET_ID 3d_fullres 3
nnUNetv2_train DATASET_ID 3d_fullres 4
```

### 2. Knowledge Distillation Training

Use the trained teacher models for knowledge distillation:

```bash
# Basic usage (using all available folds of teacher models)
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2

# Advanced usage - specify specific teacher model folds
nnUNetv2_distillation_train -d DATASET_ID -f 0 -tf 0 1 2 3 4 -a 0.3 -temp 3.0 -r 2

# Continue previous training
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2 -c_continue
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

### 3. Export ONNX Model

Export the trained student model to ONNX format for fast inference:

```bash
# Basic export
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2

# Verbose information display
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -v

# Specify output path
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -o /path/to/output.onnx
```

Parameter description:
- `-d, --dataset_id`: Dataset ID
- `-f, --fold`: Model fold number
- `-r, --reduction_factor`: Feature reduction factor, must be consistent with training
- `-o, --output`: Output ONNX file path
- `-v, --verbose`: Display detailed information
- `-static`: Use static input shape (default uses dynamic shape)

## Parameter Tuning Recommendations

For the best balance between performance and accuracy, the following parameter settings are recommended:

1. **Feature Reduction Factor (`-r`)**:
   - 2: Balanced mode, model size reduced by 75%, minimal accuracy loss
   - 4: Lightweight mode, model size reduced by 94%, may have 1-2% accuracy loss
   - 8: Ultra-lightweight mode, model size reduced by 98%, suitable for extremely resource-constrained scenarios

2. **Distillation Loss Weight (`-a`)**:
   - 0.3: Default value, suitable for most scenarios
   - 0.5: More emphasis on soft label knowledge, suitable for complex segmentation tasks
   - 0.1: More emphasis on hard labels, suitable for simple segmentation tasks

3. **Distillation Temperature (`-temp`)**:
   - 3.0: Default value, provides moderately smooth soft labels
   - 1.0: Sharper soft labels, closer to original predictions
   - 5.0: Very smooth soft labels, maximizes knowledge transfer

## Performance Comparison

The table below shows the comparison of student model performance with different feature reduction factors compared to the original nnUNet:

| Reduction Factor | Parameter Reduction | Memory Usage Reduction | Inference Speed Improvement | Dice Coefficient Change |
|------------|----------|------------|-----------|------------|
| r=2        | 75%      | 70%        | ~20x     | <0.5%      |
| r=4        | 94%      | 88%        | ~35x     | 1-2%       |
| r=8        | 98%      | 95%        | ~50x     | 2-5%       |

## Advanced Usage

### Using Custom Teacher Models

If you have teacher models trained elsewhere, you can specify them using the `-t` parameter:

```bash
nnUNetv2_distillation_train -d DATASET_ID -f 0 -t /path/to/teacher/model -a 0.3 -temp 3.0 -r 2
```

### Multi-GPU Training

Although student models are usually smaller, multi-GPU training is still supported:

```bash
CUDA_VISIBLE_DEVICES=0,1 nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2
```

## Troubleshooting

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

## Citation

If you use FastnnUNet Knowledge Distillation in your research, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
``` 