# FastnnUNet

FastnnUNet是一个基于nnUNetv2架构的高性能医学图像分割框架。通过结合知识蒸馏技术，它实现了与原始nnUNet相同的准确性，但推理性能提高了数十倍。

> ⚠️ **重要说明**: 完整的FastnnUNet包含FastnnUNet代码的C++版本，但目前在此存储库中暂不提供，尽管已经过工程化和测试的稳定版本没有问题。
> 
> 🔍 **当前发布状态**: 目前仅发布了关于蒸馏模块的代码，由于当前正在进行的项目需求，其他模块的代码暂时不可用。欢迎使用FastnnUNet的开发者自行实现剩余的两个模块，这充满了挑战。当然，也欢迎使用蒸馏模块实现FastnnUNet模型，用于训练和针对基准nnUNet模型进行测试。
>
> 📅 **更新时间**: 2025年4月25日
> 
> 🔮 **即将推出**: 其余模块将在未来陆续发布，敬请期待!
> 
> ⭐ 如果它对您的工作或研究有所帮助，请随时为FastnnUNet添加星标！如果您遇到任何相关问题或不足之处，请联系我，持续改进是我个人最大的初衷！^-^

## Project Background and Objectives

FastnnUNet aims to address two major deficiencies of the original nnUNet framework:
1. **Slow inference speed**: The original nnUNet, despite its high accuracy, has slow inference speed, making it difficult to meet real-time clinical application requirements
2. **Deployment challenges**: Large model size, high computational resource requirements, difficult to deploy and migrate in resource-constrained environments

Through knowledge distillation technology, FastnnUNet successfully solves these problems while maintaining segmentation accuracy comparable to the original nnUNet. The model supports multiple deployment formats:
- **PyTorch**: Supports native PyTorch format, suitable for research environments
- **ONNX**: Supports ONNX format export, providing cross-platform compatibility
- **TensorRT**: Supports high-performance TensorRT acceleration, achieving inference in seconds

## Key Features

- **Based on nnUNetv2**: Inherits nnUNetv2's powerful adaptive architecture and excellent segmentation performance
- **3D Knowledge Distillation**: Uses 5-fold cross-validation trained teacher models to guide lightweight student model learning
- **High-performance Inference**: Maintains accuracy consistent with the original nnUNet, but with inference speed improved by tens of times
- **Complete Compatibility**: Inference parameters completely consistent with the original nnUNet, supporting seamless replacement
- **Multi-format Support**: Supports PyTorch, ONNX, and TensorRT formats, adapting to different deployment scenarios
- **Lightweight Design**: Greatly reduces model parameters and computational load, suitable for edge device deployment

## Workflow

1. **Standard nnUNet Training**:
   - Use the nnUNetv2 standard process for 5-fold cross-validation training
   - Obtain high-precision but computationally intensive teacher models

2. **Knowledge Distillation**:
   - Train lightweight student models based on teacher models
   - Jointly use soft labels and hard labels for distillation
   - Maintain segmentation accuracy while significantly reducing parameter count and computational load

3. **Multi-format Export and Deployment**:
   - Support export to PyTorch, ONNX, and TensorRT formats
   - Adapt to different hardware platforms and deployment environments
   - Optimize performance for different formats

4. **Fast Inference**:
   - Use distilled lightweight models for inference
   - Fully compatible with nnUNet inference parameters
   - Performance improved by tens of times
   - Support multiple medical imaging modalities such as CT/MR
   - No fixed Patch size required, completely dependent on nnUNet preprocessing configuration
   - Achieve complete 3D image processing in seconds with TensorRT

## System Components

The system consists of three main parts:

### 1. Knowledge Distillation Module

Used for knowledge transfer from standard nnUNet models (teacher models) to lightweight models (student models). For detailed information, please refer to the [Distillation Module Documentation](./distillation/README.md).

### 2. Fast Inference Module

Performs efficient inference based on distilled lightweight models, significantly improving performance while maintaining accuracy. For detailed information, please refer to the [Inference Module Documentation](./inference/README.md).

### 3. C++ Engine Module

A high-performance C++ implementation of FastnnUNet built on CUDA operators and TensorRT for production-level deployment. This engine enables ultra-fast inference (seconds) for CT and MRI images in clinical settings. For detailed information, please refer to the [C++ Engine Documentation](./engine/README.md).

## Usage Instructions

### 1. Data Preparation and Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 2. Teacher Model Training (Standard nnUNet 5-fold Cross-validation)

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0
nnUNetv2_train DATASET_ID 3d_fullres 1
nnUNetv2_train DATASET_ID 3d_fullres 2
nnUNetv2_train DATASET_ID 3d_fullres 3
nnUNetv2_train DATASET_ID 3d_fullres 4
```

### 3. Knowledge Distillation (Training Lightweight Student Models)

```bash
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2
```

Parameter explanation: `-r` represents the reduction factor of the student model's size, 2 means half the size of the original nnUNet. Through testing, even with 1/6 of the size (`-r 6`), the FastnnUNet model can achieve segmentation accuracy close to the original nnUNet.

### 4. Export Different Model Formats

```bash
# Export ONNX format (for cross-platform deployment)
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -v

# Convert to TensorRT format (for ultimate performance optimization, requires additional steps)
# Please refer to TensorRT documentation to convert ONNX models to TensorRT engines
```

### 5. Fast Inference

Supports multiple inference methods:
- **PyTorch Inference**: Suitable for research environments, highly flexible
- **ONNX Inference**: Cross-platform compatible, simple deployment
- **TensorRT Inference**: Based on CUDA operators, optimal performance, can achieve real-time performance in engineering deployments

Application scenarios: Can be used for assisted interventional diagnosis, intraoperative real-time guidance, or rapid analysis and processing of large-scale medical imaging data.

## Performance Comparison

The table below shows the comparison between FastnnUNet and the original nnUNet on different metrics:

| Model | Segmentation Accuracy (Dice) | Inference Time | Parameter Count | Memory Usage | Deployment Difficulty |
|------|---------------|---------|-------|---------|---------|
| Original nnUNet | Baseline | Baseline | Baseline | Baseline | Complex |
| FastnnUNet (PyTorch) | Comparable | 5-10x improvement | 75-95% reduction | 70% reduction | Medium |
| FastnnUNet (ONNX) | Comparable | 10-20x improvement | 75-95% reduction | 80% reduction | Simple |
| FastnnUNet (TensorRT) | Comparable | 20-50x improvement | 75-95% reduction | 85% reduction | Medium |
| FastnnUNet (C++/TensorRT) | Comparable | 30-60x improvement | 75-95% reduction | 85% reduction | Production-ready |

## Citation

If you use FastnnUNet, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
``` 