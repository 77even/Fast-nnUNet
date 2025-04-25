# FastnnUNet 🔬

<div align="center">
  <img src="./public/fastnnunet-logo.png" alt="FastnnUNet Logo" width="1000">
</div>

FastnnUNet is a high-performance medical image segmentation framework based on the nnUNetv2 architecture. By combining knowledge distillation techniques, it achieves the same accuracy as the original nnUNet but with inference performance improved by tens of times. This project was developed by the AI Lab of Xiaozhi Future (Chengdu) Tech Inc.

> ⚠️ **Important Note**: The full FastnnUNet includes a C++ version of the FastnnUNet code, but it is not available for this repository at this time, even though the stable version that has been engineered and tested has no problems.
> 
> 🔍 **Current Release Status**: Currently only the code about the distillation module is released, the code for the other modules is temporarily unavailable due to the needs of the current ongoing project. Developers using FastnnUNet are welcome to implement the remaining two modules on their own, which is full of challenges. Of course, feel free to use the distillation module to implement FastnnUNet models for training and testing against benchmark nnUNet models.
>
> 🚀 **Update Date**: April 25, 2025
> 
> 🚀 **Coming Soon**: The remaining modules will be released successively in the future, stay tuned!
> 
> ⭐ If it is helpful to your study research, please feel free to add FastnnUNet's star mark! Of course, if you encounter any related problems or deficiencies, please contact us, and continuous improvement is our greatest original intention! ^-^
> 
> <h3>❗❗ <strong>Attention</strong>: No commercial use is allowed</h3>

## 🎯 Project Background and Objectives

FastnnUNet aims to address two major deficiencies of the original nnUNet framework:
1. ⏱️ **Slow inference speed**: The original nnUNet, despite its high accuracy, has slow inference speed, making it difficult to meet real-time clinical application requirements
2. 🖥️ **Deployment challenges**: Large model size, high computational resource requirements, difficult to deploy and migrate in resource-constrained environments

Through a new 3D probability map knowledge distillation technique derived from the native nnUNet framework, FastnnUNet successfully solves these problems while maintaining segmentation accuracy comparable to the original nnUNet. The model supports multiple deployment formats:
- 🔥 **PyTorch**: Supports native PyTorch format, suitable for research environments
- 🌐 **ONNX**: Supports ONNX format export, providing cross-platform compatibility
- ⚡ **TensorRT**: Supports high-performance TensorRT acceleration, achieving inference in seconds

## ✨ Key Features

- 🧠 **Based on nnUNetv2**: Inherits nnUNetv2's powerful adaptive architecture and excellent segmentation performance
- 🔄 **3D Probability Map Knowledge Distillation**: Uses 5-fold cross-validation trained teacher models to guide lightweight student model learning
- 🚀 **High-performance Inference**: Maintains accuracy consistent with the original nnUNet, but with inference speed improved by tens of times
- 🔄 **Complete Compatibility**: Inference parameters completely consistent with the original nnUNet, supporting seamless replacement
- 🔌 **Multi-format Support**: Supports PyTorch, ONNX, and TensorRT formats, adapting to different deployment scenarios
- 🪶 **Lightweight Design**: Greatly reduces model parameters and computational load, suitable for edge device deployment

## 📊 Workflow

1. 📚 **Standard nnUNet Training**:
   - Use the nnUNetv2 standard process for 5-fold cross-validation training
   - Obtain high-precision but computationally intensive teacher models

2. 🧪 **Knowledge Distillation**:
   - Train lightweight student models based on teacher models
   - Jointly use soft labels and hard labels for distillation
   - Maintain segmentation accuracy while significantly reducing parameter count and computational load

3. 📦 **Multi-format Export and Deployment**:
   - Support export to PyTorch, ONNX, and TensorRT formats
   - Adapt to different hardware platforms and deployment environments
   - Optimize performance for different formats

4. ⚡ **Fast Inference**:
   - Use distilled lightweight models for inference
   - Fully compatible with nnUNet inference parameters
   - Performance improved by tens of times
   - Support multiple medical imaging modalities such as CT/MR
   - No fixed Patch size required, completely dependent on nnUNet preprocessing configuration
   - Achieve complete 3D image processing in seconds with TensorRT

## 🧩 System Components

The system consists of three main parts:

### 1. 🔮 Knowledge Distillation Module

Used for knowledge transfer from standard nnUNet models (teacher models) to lightweight models (student models). For detailed information, please refer to the [Distillation Module Documentation](./distillation/README.md).

### 2. 🚀 Fast Inference Module

Performs efficient inference based on distilled lightweight models, significantly improving performance while maintaining accuracy. For detailed information, please refer to the [Inference Module Documentation](./inference/README.md).

### 3. ⚙️ C++ Engine Module

A high-performance C++ implementation of FastnnUNet built on CUDA operators and TensorRT for production-level deployment. This engine enables ultra-fast inference (seconds) for CT and MRI images in clinical settings. For detailed information, please refer to the [C++ Engine Documentation](./engine/README.md).

## 📋 Usage Instructions

### 1. 📥 Data Preparation and Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### 2. 🏫 Teacher Model Training (Standard nnUNet 5-fold Cross-validation)

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0
nnUNetv2_train DATASET_ID 3d_fullres 1
nnUNetv2_train DATASET_ID 3d_fullres 2
nnUNetv2_train DATASET_ID 3d_fullres 3
nnUNetv2_train DATASET_ID 3d_fullres 4
```

### 3. 🧠 Knowledge Distillation (Training Lightweight Student Models)

```bash
nnUNetv2_distillation_train -d DATASET_ID -f 0 -a 0.3 -temp 3.0 -r 2
```

Parameter explanation: `-r` represents the reduction factor of the student model's size, 2 means half the size of the original nnUNet. Through testing, even with 1/6 of the size (`-r 6`), the FastnnUNet model can achieve segmentation accuracy close to the original nnUNet.

### 4. 📤 Export Different Model Formats

```bash
# Export ONNX format (for cross-platform deployment)
nnUNetv2_distillation_export_onnx -d DATASET_ID -f 0 -r 2 -v

# Convert to TensorRT format (for ultimate performance optimization, requires additional steps)
# Please refer to TensorRT documentation to convert ONNX models to TensorRT engines
```

### 5. ⚡ Fast Inference

Supports multiple inference methods:
- 🔬 **PyTorch Inference**: Suitable for research environments, highly flexible
- 🌐 **ONNX Inference**: Cross-platform compatible, simple deployment
- 🚀 **TensorRT Inference**: Based on CUDA operators, optimal performance, can achieve real-time performance in engineering deployments

Application scenarios: Can be used for assisted interventional diagnosis, intraoperative real-time guidance, or rapid analysis and processing of large-scale medical imaging data.

## 📈 Performance Comparison

The table below shows the comparison between FastnnUNet and the original nnUNet on different metrics:

| Model | Segmentation Accuracy (Dice) | Inference Time | Parameter Count | Memory Usage | Deployment Difficulty |
|------|---------------|---------|-------|---------|---------|
| Original nnUNet | Baseline | Baseline | Baseline | Baseline | Complex |
| FastnnUNet (PyTorch) | Comparable | 5-10x improvement | 75-95% reduction | 70% reduction | Medium |
| FastnnUNet (ONNX) | Comparable | 10-20x improvement | 75-95% reduction | 80% reduction | Simple |
| FastnnUNet (TensorRT) | Comparable | 20-50x improvement | 75-95% reduction | 85% reduction | Medium |
| FastnnUNet (C++/TensorRT) | Comparable | 30-60x improvement | 75-95% reduction | 85% reduction | Production-ready |

## 📚 Project Status

| Module | Status | Expected Release |
|--------|--------|-----------------|
| 🔮 Knowledge Distillation | ✅ Released | Available Now |
| 🚀 Fast Inference | 🔜 Planned | Coming Soon |
| ⚙️ C++ Engine | 🔜 Planned | Coming Soon |

## 📝 Citation

FastnnUNet is overwhelmingly derived from nnUNet, If you use FastnnUNet, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
``` 
