# FastnnUNet C++ Engine

The FastnnUNet C++ Engine is a high-performance implementation of the FastnnUNet architecture designed for production-level deployment. Built on CUDA operators and TensorRT, this engine provides ultra-fast inference for CT and MRI images, completing the entire inference pipeline in seconds.

## Key Features

- **High Performance**: Optimized C++ implementation with CUDA operators and TensorRT acceleration
- **Production Ready**: Engineered for stable clinical deployment with robust error handling
- **Fast Inference**: Complete inference pipeline for 3D medical images in seconds
- **Memory Efficient**: Optimized memory management for handling large volumetric data
- **Platform Compatible**: Designed to work on various NVIDIA GPU platforms
- **Clinical Integration**: Easy integration with clinical workflows and DICOM systems

## Technical Specifications

- **Implementation**: C++17 with CUDA 11.x/12.x support
- **Acceleration**: TensorRT optimization with FP16/INT8 quantization options
- **Dependencies**: CUDA, cuDNN, TensorRT, OpenCV (minimal)
- **GPU Support**: NVIDIA GPUs with compute capability 6.0+
- **Input Formats**: Supports NIfTI, DICOM, and raw data formats
- **Integration**: C++ API available

## Performance Metrics

| Image Type | Size | Original nnUNet (PyTorch) | FastnnUNet (PyTorch) | FastnnUNet (C++/TensorRT) |
|------------|------|--------------------------|---------------------|--------------------------|
| Brain MRI  | 256×256×160 | 12-15s | 4-8s | 0.3-0.5s |
| Chest CT   | 512×512×400 | 30-40s | 14-16s | 0.1-0.5s |
| Abdominal CT | 512×512×500+ | 40-60s | 15-18s | 4.5-7.5s |

*All metrics measured on NVIDIA RTX 5070Ti 4090 4080 3080 3060 3050 2080Ti GPUs

## Usage

### C++ API Example

```cpp
#include "fastnnunet/engine.h"

// Initialize engine with model path
FastnnUNet::Engine engine("path/to/trt/model");

// Load and preprocess image
auto image = engine.loadImage("patient_ct.nii.gz/.nii");

// Run inference
auto segmentation = engine.infer(image);

// Save result
engine.saveSegmentation(segmentation, "output_segmentation.nii.gz");
```


Requirements:
- CUDA 11.x or newer
- TensorRT 8.x or newer
- CMake 3.18+
- GCC 7+ or MSVC 2022+

```bash
# Clone repository
git clone https://github.com/username/FastnnUNet.git
cd FastnnUNet/engine

# Build
mkdir build && cd build
cmake ..
make -j8

# Install
make install
```

## Converting Models

The C++ engine uses optimized TensorRT models. Convert your ONNX models using:

```bash

# From ONNX
trtexec --onnx path/to/model.onnx --saveEngine engine.trt --fp16 --shapes=input:batch_size x 1 x D x H x W(enable batch infer)
```

## License

This component follows the same license as the main FastnnUNet project.

## Citation

If you use the FastnnUNet C++ Engine in your research, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). 
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. 
Nature methods, 18(2), 203-211.
``` 