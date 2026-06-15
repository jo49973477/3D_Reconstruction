import argparse
import torch
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from network.net import NeRF_MLP

def extract_mesh(args):
    print(f"Loading NeRF model from {args.checkpoint}...")
    
    # 1. Load the trained NeRF model (adjust the model class to match your specific architecture)
    model = NeRF_MLP()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    
    # Move model to GPU for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Generating voxel grid with resolution {args.grid_res}^3...")
    
    # 2. Set up the 3D bounding box and generate the voxel grid
    N = args.grid_res
    bound = args.bound
    x = torch.linspace(-bound, bound, N)
    y = torch.linspace(-bound, bound, N)
    z = torch.linspace(-bound, bound, N)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

    # Flatten the coordinates to shape (N*N*N, 3)
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)

    print("Querying MLP for density values (Processing in chunks to prevent OOM)...")
    
    # 3. Query the MLP to extract density (sigma) values
    densities = torch.zeros(N * N * N, device='cpu')
    chunk_size = args.chunk_size

    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk_size):
            chunk_coords = coords[i:i+chunk_size]
            
            # View directions are not required for geometry extraction.
            # We only extract the density (sigma) output from the model.
            _, chunk_sigma = model(chunk_coords) 
            
            densities[i:i+chunk_size] = chunk_sigma.cpu()

    # Reshape back to the 3D grid format
    densities = densities.reshape(N, N, N).numpy()

    print(f"Running Marching Cubes algorithm with threshold {args.threshold}...")
    
    # 4. Extract the surface mesh using the Marching Cubes algorithm
    verts, faces, normals, values = marching_cubes(densities, level=args.threshold)

    # 5. Normalize voxel coordinates (0 to N-1) back to the original 3D spatial coordinates
    verts = verts / (N - 1) * (2.0 * bound) - bound

    print(f"Exporting mesh to {args.output}...")
    
    # 6. Export as a PLY file using Trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(args.output)

    print("Extraction complete! You can view the output mesh in standard 3D viewers.")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract a 3D mesh from a trained NeRF model using Marching Cubes")
    
    parser.add_argument("--checkpoint", type=str, default="nerf_weights.pt", 
                        help="Path to the trained NeRF checkpoint")
    parser.add_argument("--output", type=str, default="nerf_extracted_mesh.ply", 
                        help="Path to save the output PLY file")
    parser.add_argument("--grid-res", type=int, default=256, 
                        help="Resolution of the voxel grid (N x N x N). Reduce if experiencing OOM errors.")
    parser.add_argument("--bound", type=float, default=1.0, 
                        help="Bounding box limits (e.g., setting 1.0 means limits from -1.0 to 1.0)")
    parser.add_argument("--chunk-size", type=int, default=1024 * 64, 
                        help="Number of coordinates to process per batch (prevents VRAM overflow)")
    parser.add_argument("--threshold", type=float, default=15.0, 
                        help="Density threshold (iso-level) for surface extraction. Requires empirical tuning.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_mesh(args)