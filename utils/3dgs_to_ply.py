import argparse
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def convert_to_universal_ply(pt_path, ply_path):
    print(f"🚀 Starting data conversion to universal viewer compatible mode...")
    
    # 1. Load the file
    checkpoint = torch.load(pt_path, map_location='cpu')
    means = checkpoint['means'].numpy()
    
    # 2. Restore raw color data (Logit) to 0~1 range
    colors_logit = checkpoint['colors']
    colors_rgb = torch.sigmoid(colors_logit).numpy()
    
    # 3. Strictly convert 0~1 values to 0~255 integers (uint8)
    colors_uint8 = (colors_rgb * 255.0).clip(0, 255).astype(np.uint8)
    
    # 4. Use the simplest and most intuitive universal labels
    dtype_universal = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1') # u1 = 0~255 integer
    ]
    
    elements = np.empty(means.shape[0], dtype=dtype_universal)
    
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    elements['red'] = colors_uint8[:, 0]
    elements['green'] = colors_uint8[:, 1]
    elements['blue'] = colors_uint8[:, 2]
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)
    print(f"✅ {ply_path} generation complete! Colors should now be visible in any viewer.")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert 3DGS .pt checkpoint to universal .ply format")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default="3dgs_checkpoint_tless/tless_latest.pt", 
        help="Path to the input .pt file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="universal_output.ply", 
        help="Path to the output .ply file"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_to_universal_ply(args.input, args.output)