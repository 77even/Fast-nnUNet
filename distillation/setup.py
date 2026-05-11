from setuptools import setup

setup(
    name="nnunetv2_distillation",
    version="1.3.0",
    install_requires=[
        "torch>=1.6.0",
        "nnunetv2>=2.5,<3.0",
    ],
    entry_points={
        'console_scripts': [
            'nnUNetv2_distillation_train=fast_nnunet_distillation_train:main',
            'nnUNetv2_resenc_distillation_train=fast_nnunet_resenc_distillation_train:main',
            'nnUNetv2_primus_distillation_train=fast_nnunet_primus_distillation_train:main',
            'nnUNetv2_distillation_export_onnx=fast_nnunet_distillation_export_onnx:main',
            'nnUNetv2_resenc_distillation_export_onnx=fast_nnunet_resenc_distillation_export_onnx:main',
            'nnUNetv2_primus_distillation_export_onnx=fast_nnunet_primus_distillation_export_onnx:main',
        ],
    },
    py_modules=[
        'nnunet_distillation_trainer',
        'fast_nnunet_distillation_train',
        'fast_nnunet_resenc_distillation_train',
        'fast_nnunet_primus_distillation_train',
        'fast_nnunet_distillation_export_onnx',
        'fast_nnunet_resenc_distillation_export_onnx',
        'fast_nnunet_primus_distillation_export_onnx',
        'primus_distillation_trainer',
    ],
    python_requires='>=3.7',
    author="Justin",
    author_email="ljq122377@gmail.com",
    description="Knowledge distillation module for FastnnUNet based on nnUNetV2",
    keywords="deep learning, segmentation, knowledge distillation, nnUNet",
)

