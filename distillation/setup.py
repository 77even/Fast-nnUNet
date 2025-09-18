from setuptools import setup, find_packages

setup(
    name="nnunetv2_distillation",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "nnunetv2"
    ],
    entry_points={
        'console_scripts': [
            'nnUNetv2_distillation_train=distillation.fast_nnunet_distillation_train:main',
            'nnUNetv2_resenc_distillation_train=distillation.fast_nnunet_resenc_distillation_train:main',
            'nnUNetv2_distillation_export_onnx=distillation.fast_nnunet_distillation_export_onnx:main',
            'nnUNetv2_resenc_distillation_export_onnx=distillation.fast_nnunet_resenc_distillation_export_onnx:main',
        ],
    },
    py_modules=[
        'distillation.fast_nnunet_distillation_train',
        'distillation.fast_nnunet_resenc_distillation_train',
        'distillation.fast_nnunet_distillation_export_onnx',
        'distillation.fast_nnunet_resenc_distillation_export_onnx'
    ],
    python_requires='>=3.7',
    author="Justin",
    author_email="ljq122377@gmail.com",
    description="Knowledge distillation module for FastnnUNet based on nnUNetV2",
    keywords="deep learning, segmentation, knowledge distillation, nnUNet",
) 
