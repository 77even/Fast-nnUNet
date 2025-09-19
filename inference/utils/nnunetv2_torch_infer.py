from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
from multiprocessing import freeze_support

def main():
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir='/path/to/your/model/folder',
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    predictor.predict_from_files(
        '/path/to/your/input/folder',
        '/path/to/your/output/folder',
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=16,
        num_processes_segmentation_export=16
    )

if __name__ == '__main__':
    freeze_support() 
    main()