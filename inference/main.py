#!/usr/bin/env python3
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Title: Fast-nnUNet: High-performance medical image segmentation framework based on the nnUNetv2 architecture
## Authors: Justin Lee
## Description: FastnnUNet main

"""
Fast-nnUNet inference
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.api import FastnnUNetInferencer, VTKModelGenerator
from inference.api.rest_api import FastnnUNetAPI


def setup_logging(level: str = "INFO"):
    """Setup logging"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_predict_single(args):
    """Single image inference command"""
    try:
        # Create inferenceer
        inferencer = FastnnUNetInferencer(
            config_path=args.config,
            model_path=args.model
        )
        
        # Load model
        inferencer.load_model()
        
        # Execute inference
        segmentation, output_path = inferencer.predict_single_image(
            image_path=args.input,
            output_path=args.output,
            save_probabilities=args.save_probabilities
        )
        
        print(f"✅ Inference completed!")
        print(f"   - Segmentation result shape: {segmentation.shape}")
        print(f"   - Unique labels: {sorted(set(segmentation.flatten()))}")
        print(f"   - Result saved to: {output_path}")
        
        # Generate VTK model (if needed)
        if args.generate_vtk:
            vtk_generator = VTKModelGenerator(color_file_path=args.color_file)
            vtk_output = str(Path(args.output).with_suffix('.vtk'))
            
            vtk_path = vtk_generator.generate_vtk_model(
                mask_path=output_path,
                output_path=vtk_output,
                smoothing_factor=args.smoothing_factor,
                decimation_factor=args.decimation_factor
            )
            
            print(f"   - VTK model saved to: {vtk_path}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return 1
    
    return 0


def cmd_predict_batch(args):
    """Batch inference command"""
    try:
        # Create inferenceer
        inferencer = FastnnUNetInferencer(
            config_path=args.config,
            model_path=args.model
        )
        
        # Load model
        inferencer.load_model()
        
        # Batch inference
        results = inferencer.predict_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_probabilities=args.save_probabilities,
            file_pattern=args.pattern
        )
        
        print(f"✅ Batch inference completed!")
        print(f"   - Successfully processed {len(results)} files")
        
        # Batch generate VTK model (if needed)
        if args.generate_vtk:
            vtk_generator = VTKModelGenerator(color_file_path=args.color_file)
            vtk_output_dir = Path(args.output_dir) / "vtk_models"
            vtk_output_dir.mkdir(exist_ok=True)
            
            success_count = 0
            for input_file, seg_file in results.items():
                try:
                    input_name = Path(input_file).stem
                    if input_name.endswith('.nii'):
                        input_name = input_name[:-4]
                    
                    vtk_output_path = vtk_output_dir / f"{input_name}_model.vtk"
                    
                    vtk_generator.generate_vtk_model(
                        mask_path=seg_file,
                        output_path=str(vtk_output_path),
                        smoothing_factor=args.smoothing_factor,
                        decimation_factor=args.decimation_factor
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Generate VTK model failed {input_file}: {e}")
            
            print(f"   - Successfully generated {success_count} VTK models")
        
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return 1
    
    return 0


def cmd_serve_api(args):
    """Start API server command"""
    try:
        api = FastnnUNetAPI(
            config_path=args.config,
            model_path=args.model
        )
        
        # Initialize VTK generator
        if args.color_file:
            api.initialize_vtk_generator(args.color_file)
        
        print(f"🚀 Start API server: http://{args.host}:{args.port}")
        api.run(host=args.host, port=args.port, debug=args.debug)
        
    except Exception as e:
        print(f"❌ API server start failed: {e}")
        return 1
    
    return 0


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fast-nnUNet High-performance medical image segmentation inference tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Single image inference
  python inference/main.py predict-single \\
      --input image.nii.gz \\
      --output segmentation.nii.gz \\
      --config config.json \\
      --model model.onnx

  # Batch inference
  python inference/main.py predict-batch \\
      --input-dir /path/to/images \\
      --output-dir /path/to/results \\
      --config config.json \\
      --model model.onnx

  # Start API server
  python inference/main.py serve-api \\
      --config config.json \\
      --model model.onnx \\
      --host 0.0.0.0 \\
      --port 5000
        """
    )
    
    # Global parameters
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single image inference command
    predict_single_parser = subparsers.add_parser('predict-single', help='Single image inference')
    predict_single_parser.add_argument('--input', '-i', required=True, help='Input image path')
    predict_single_parser.add_argument('--output', '-o', required=True, help='Output segmentation result path')
    predict_single_parser.add_argument('--config', '-c', help='Configuration file path')
    predict_single_parser.add_argument('--model', '-m', help='Model file path')
    predict_single_parser.add_argument('--save-probabilities', action='store_true', help='Save probabilities')
    predict_single_parser.add_argument('--generate-vtk', action='store_true', help='Generate VTK model')
    predict_single_parser.add_argument('--color-file', help='VTK color table file path')
    predict_single_parser.add_argument('--smoothing-factor', type=float, default=0.5, help='VTK smoothing factor')
    predict_single_parser.add_argument('--decimation-factor', type=float, default=0.2, help='VTK decimation factor')
    
    # Batch inference command
    predict_batch_parser = subparsers.add_parser('predict-batch', help='Batch inference')
    predict_batch_parser.add_argument('--input-dir', required=True, help='Input image directory')
    predict_batch_parser.add_argument('--output-dir', required=True, help='Output result directory')
    predict_batch_parser.add_argument('--config', '-c', help='Configuration file path')
    predict_batch_parser.add_argument('--model', '-m', help='Model file path')
    predict_batch_parser.add_argument('--pattern', default='*.nii.gz', help='File matching pattern')
    predict_batch_parser.add_argument('--save-probabilities', action='store_true', help='Save probabilities')
    predict_batch_parser.add_argument('--generate-vtk', action='store_true', help='Generate VTK model')
    predict_batch_parser.add_argument('--color-file', help='VTK color table file path')
    predict_batch_parser.add_argument('--smoothing-factor', type=float, default=0.5, help='VTK smoothing factor')
    predict_batch_parser.add_argument('--decimation-factor', type=float, default=0.2, help='VTK decimation factor')
    
    # API server command
    serve_api_parser = subparsers.add_parser('serve-api', help='Start API server')
    serve_api_parser.add_argument('--config', '-c', help='Configuration file path')
    serve_api_parser.add_argument('--model', '-m', help='Model file path')
    serve_api_parser.add_argument('--color-file', help='VTK color table file path')
    serve_api_parser.add_argument('--host', default='0.0.0.0', help='Server host address')
    serve_api_parser.add_argument('--port', type=int, default=5000, help='Server port')
    serve_api_parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # Set logging
    setup_logging(args.log_level)
    
    # Check command
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute corresponding command
    if args.command == 'predict-single':
        return cmd_predict_single(args)
    elif args.command == 'predict-batch':
        return cmd_predict_batch(args)
    elif args.command == 'serve-api':
        return cmd_serve_api(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    exit(main())
