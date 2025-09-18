<div align="center">
  <img src="../docs/assets/fastnnunet-logo.png" alt="FastnnUNet Logo" width="600">
</div>

# 🚀 FastnnUNet Fast Inference Module

The FastnnUNet Inference Module is a component for high-performance medical image segmentation based on knowledge distillation-trained lightweight models. This module maintains accuracy and preprocessing/postprocessing workflows identical to the original nnUNet, but with inference speed improved by tens of times.

## ✨ Features

- ⚡ **High-speed Inference**: Based on distilled lightweight models, performance improves by tens of times compared to the original nnUNet
- 🎯 **Consistent Accuracy**: Segmentation accuracy comparable to the original nnUNet, without sacrificing accuracy
- 💾 **GPU Memory Friendly**: Significantly reduces GPU memory requirements, supports running on mid to low-end graphics cards
- 🔄 **ONNX Optimization**: Utilizes ONNX Runtime optimization for further performance improvements
- 🏥 **Multiple Anatomical Regions**: Supports segmentation of multiple anatomical regions including skeletal, thoracic, abdominal, head and neck, pelvic, spine, limbs, and whole body
- 🧮 **Advanced Postprocessing**: Specialized postprocessing algorithms for different anatomical structures to improve segmentation quality

## 🔍 Supported Anatomical Structures

The FastnnUNet Inference Module supports segmentation of the following anatomical regions:

1. 🦴 **Skeletal System** (different precision levels):
   - Low-precision model (Low): Fast skeletal segmentation
   - Medium-precision model (Medium): More detailed skeletal structure segmentation
   - High-precision model (High): Complete detailed skeletal segmentation
   - Cortical bone segmentation: Optional detailed cortical bone segmentation

2. 🫀 **Visceral Organs**:
   - Thoracic organ segmentation
   - Abdominal organ segmentation (standard and extended versions)
   - Pelvic region segmentation

3. 🧠 **Specialized Regions**:
   - Head and neck segmentation (9-region and 30-region versions)
   - Spine segmentation
   - Limb segmentation

4. 👤 **Whole Body Segmentation**:
   - Complete whole-body multi-organ segmentation

## ⚙️ Configuration Description

Same as the original nnUNet, FastnnUNet supports the following configurations:

###### 2D Models:

    1. Operate only on 2D slices, treating 3D volume data as a series of independent 2D slices
    2. Low computational resource requirements, fast training speed
    3. Suitable for tasks where inter-slice information is not very important

######  3D_fullres:

    1. Operate directly on full-resolution 3D volumes
    2. Can capture spatial information in all three dimensions
    3. Lower computational cost than the original nnUNet, but maintains spatial context information
    4. Suitable for tasks requiring complete spatial context

######  3D_lowres:

    1. Operate on downsampled 3D volumes
    2. Reduces computational resource requirements
    3. Sacrifices some detail, but preserves global 3D spatial information
    4. Often used as the first stage in cascade methods

######  3D_cascade_fullres:

    1. Two-stage method, combining the advantages of 3D_lowres and 3D_fullres
    2. First stage: Uses 3D_lowres model to obtain rough segmentation
    3. Second stage: Based on first stage results, 3D_fullres model refines at full resolution
    4. Able to capture both global context and local details simultaneously
    5. Suitable for processing high-resolution or large-volume medical images
    6. Compared to the original nnUNet, both stages get performance improvements

## 📋 Usage

### 🔰 Basic Usage

FastnnUNet provides a simple and easy-to-use command line interface:

```bash
python inference/fastnnunet.py --input INPUT_IMAGE_PATH --mode ANATOMICAL_REGION [--enable_subseg]
```

Parameter description:
- `--input`: Input CT image path (supports nii/nii.gz formats)
- `--mode`: Anatomical region, options include:
  - `bone`: Skeletal segmentation
  - `chest`: Thoracic segmentation
  - `abdomen`: Abdominal segmentation
  - `headneck`: Head and neck segmentation
  - `pelvic`: Pelvic segmentation
  - `spine`: Spine segmentation
  - `total`: Whole body segmentation
  - `limb`: Limb segmentation
- `--enable_subseg`: Enable detailed cortical bone segmentation (only valid in bone mode)

### 📝 Example Commands

```bash
# Skeletal segmentation
python inference/fastnnunet.py --input img/ct_image.nii.gz --mode bone

# Skeletal segmentation (including cortical bone)
python inference/fastnnunet.py --input img/ct_image.nii.gz --mode bone --enable_subseg

# Thoracic organ segmentation
python inference/fastnnunet.py --input img/chest_ct.nii.gz --mode chest

# Abdominal organ segmentation
python inference/fastnnunet.py --input img/abdomen_ct.nii.gz --mode abdomen

# Whole body segmentation
python inference/fastnnunet.py --input img/whole_body.nii.gz --mode total
```

### 🚀 Advanced Inference Features

FastnnUNet is optimized with ONNX runtime and supports the following advanced inference features:

1. 🤖 **Automatic Model Selection**: Automatically selects the best model based on anatomical region
2. 🔁 **Sliding Window Inference**: Enabled by default, can process inputs of arbitrary size
3. 🪞 **Mirror Enhancement**: Improves segmentation quality through mirroring augmentation
4. 🧵 **Multi-threading Acceleration**: Automatically utilizes multi-core CPUs for pre- and post-processing

## 🔄 Inference Pipeline

The FastnnUNet inference pipeline includes the following steps:

1. 📥 **Image Loading and Preprocessing**:
   - Read medical images and standardize orientation (LPS direction)
   - Resample to target resolution according to model configuration
   - Apply windowing and normalization

2. 🧠 **Optimized Inference**:
   - Use ONNX Runtime for efficient inference
   - Apply sliding window technique to process large volume data
   - Selectively use mirror enhancement to improve accuracy

3. 🔧 **Specialized Postprocessing**:
   - Apply specialized postprocessing algorithms for different anatomical structures
   - Connected component analysis and morphological operations
   - Small region filtering and hole filling

4. 💾 **Result Saving**:
   - Automatically save to the results directory
   - Maintain the spatial information and orientation of the original image

## 📊 Performance Comparison

The table below shows the performance comparison between FastnnUNet and the original nnUNet on typical medical image segmentation tasks:

| Model | Segmentation Accuracy (Dice) | Inference Time | GPU Memory Usage | Parameter Count |
|------|---------------|---------|---------|-------|
| Original nnUNet | Baseline | Baseline | Baseline | Baseline |
| FastnnUNet | Comparable | 20-50x improvement | 70% reduction | 80% reduction |

## 💻 System Requirements

- CUDA-compatible GPU (recommended at least 4GB VRAM)
- CUDA 11.0+
- Python 3.8+
- ONNX Runtime 1.13.1+
- SimpleITK
- NumPy, SciPy, scikit-image

## ❓ Frequently Asked Questions

1. 🤔 **Q: Are there differences in model prediction results compared to the original nnUNet?**  
   A: In the vast majority of cases, segmentation results are comparable to the original nnUNet, with Dice coefficient differences typically within 0.5%.

2. 🏎️ **Q: How can I further improve inference speed?**  
   A: For skeletal segmentation, you can use the low-precision model (bone_seg_low); disabling mirror enhancement can also improve speed.

3. 🧭 **Q: How to handle medical images with non-standard orientations?**  
   A: FastnnUNet will automatically convert input images to standard LPS orientation, no manual preprocessing required.

4. 📁 **Q: What medical image formats are supported?**  
   A: Currently supports NIfTI format (.nii and .nii.gz), with DICOM support planned for future updates.

5. ⚙️ **Q: How to change model configurations?**  
   A: Model configuration files are located in the `inference/config/3d_fullres/` directory and can be adjusted as needed.