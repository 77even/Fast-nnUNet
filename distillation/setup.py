from setuptools import setup, find_packages

setup(
    name="nnunetv2_distillation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "nnunetv2"
    ],
    entry_points={
        'console_scripts': [
            'nnUNetv2_distillation_train=nnunetv2_distillation_train:main',
            'nnUNetv2_distillation_export_onnx=nnunetv2_distillation_export_onnx:main',
        ],
    },
    python_requires='>=3.7',
    author="Justin",
    author_email="",
    description="Knowledge Distillation for FastnnUNet based on nnUNetv2",
    keywords="deep learning, segmentation, knowledge distillation, nnUNet",
) 
