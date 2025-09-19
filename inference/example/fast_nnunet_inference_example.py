#!/usr/bin/env python3
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Title: Fast-nnUNet: High-performance medical image segmentation framework based on the nnUNetv2 architecture
## Authors: Justin Lee
## Description: FastnnUNet simple inference example

import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from inference.api import FastnnUNetInferencer, VTKModelGenerator, ConfigManager


def main():
    """Main function"""
    
    # Configuration file and model path
    config_path = "inference/config/3d_fullres/sample_config.json"
    model_path = "/path/to/your/model.onnx"  # Please update to the actual model path
    
    # Input and output paths
    input_image = "/path/to/your/input/image.nii.gz"  # Please update to the actual input image path
    output_seg = "/path/to/your/output/segmentation.nii.gz"  # Output segmentation result path
    output_vtk = "/path/to/your/output/model.vtk"  # Output VTK model path
    
    print("=== Fast-nnUNet inference example ===")
    
    try:
        # 1. Create inferenceer
        print("1. Initialize inferenceer...")
        inferencer = FastnnUNetInferencer(config_path=config_path, model_path=model_path)
        
        # 2. Load model
        print("2. Load model...")
        inferencer.load_model()
        
        # 3. Display model information
        print("3. Model information:")
        model_info = inferencer.get_model_info()
        print(f"   - Input: {model_info['inputs']}")
        print(f"   - Output: {model_info['outputs']}")
        print(f"   - Execution providers: {model_info['providers']}")
        
        # 4. Execute inference
        print("4. Execute inference...")
        segmentation, output_file_path = inferencer.predict_single_image(
            image_path=input_image,
            output_path=output_seg,
            save_probabilities=False
        )
        
        print(f"   - Segmentation result shape: {segmentation.shape}")
        print(f"   - Unique labels: {list(set(segmentation.flatten()))}")
        print(f"   - Result saved to: {output_file_path}")
        
        # 5. Generate VTK model (optional)
        print("5. Generate VTK model...")
        color_file = "inference/config/vtk_colors/GenericAnatomyColors.txt"
        vtk_generator = VTKModelGenerator(color_file_path=color_file)
        
        vtk_output_path = vtk_generator.generate_vtk_model(
            mask_path=output_seg,
            output_path=output_vtk,
            smoothing_factor=0.5,
            decimation_factor=0.2
        )
        
        print(f"   - VTK model saved to: {vtk_output_path}")
        
        print("\n✅ Inference completed!")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
