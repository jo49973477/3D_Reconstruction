import argparse
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
import hydra

from config import MainConfig
from utils import *
from network.net import NeRF_MLP
from network.embedding import PositionalEncoder
from dataset import TinyNerfDataset


def render_image(model, H, W, focal, pose, cfg, device, chunk_size=4096):
    """
    Renders a 3D image from a novel camera pose.
    chunk_size: Number of rays processed simultaneously (adjust based on VRAM).
    """

    embedder_pts = PositionalEncoder(L=10)
    embedder_views = PositionalEncoder(L=4)
    
    # 1. Generate rays for the entire image resolution (H x W)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    all_rgb = [] # List to accumulate rendered pixel chunks

    for i in range(0, rays_o.shape[0], chunk_size):

        batch_rays_o = rays_o[i : i + chunk_size]
        batch_rays_d = rays_d[i : i + chunk_size]

        # --- Rendering process identical to the training step ---
        pts, z_vals = sample_points_along_rays(
            batch_rays_o, batch_rays_d, cfg.near, cfg.far, cfg.n_samples
        )
        
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[..., None, :].expand(pts.shape)
        
        # Apply positional encoding
        pts_flat = embedder_pts(pts)
        dirs_flat = embedder_views(viewdirs)
        
        # Forward pass through the MLP and perform volume rendering
        raw_outputs = model(pts_flat, dirs_flat)
        rgb_pred, _ = volume_rendering(raw_outputs, z_vals, batch_rays_d)
        
        # Append the chunk result (move to CPU to save VRAM)
        all_rgb.append(rgb_pred.cpu())

    # 3. Reassemble the scattered pixels into a single image
    # Concatenate into shape (H*W, 3), then reshape to (H, W, 3)
    final_image = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    
    # Convert PyTorch tensor (0~1) to NumPy array (0~255)
    final_image = (final_image.detach().numpy() * 255).astype(np.uint8)
    
    return final_image


def pose_spherical(theta, phi, radius):
    """
    Generates a spherical camera pose matrix.
    theta: Horizontal rotation angle (0~360 degrees)
    phi: Vertical rotation angle (Elevation/Pitch)
    radius: Distance from the origin (Zoom)
    """
    # Lambda functions to generate transformation matrices
    trans_t = lambda t: torch.tensor([ # Z-axis translation
        [1,0,0,0], [0,1,0,0], [0,0,1,t], [0,0,0,1]
    ], dtype=torch.float32)

    rot_phi = lambda phi: torch.tensor([ # X-axis rotation
        [1,0,0,0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi),  np.cos(phi), 0],
        [0,0,0,1]
    ], dtype=torch.float32)

    rot_theta = lambda th: torch.tensor([ # Y-axis rotation
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0,  np.cos(th), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Apply transformations: Translate -> Rotate Elevation -> Rotate Azimuth
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w

    # Note: Coordinate system alignment
    # Convert from PyTorch/OpenCV coordinates to OpenGL coordinates (invert X and Y)
    c2w = torch.tensor([[-1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=torch.float32) @ c2w
    
    return c2w


def parse_args():
    parser = argparse.ArgumentParser(description="Render a 360-degree video from a trained NeRF model")
    
    # Checkpoint and Data parameters
    parser.add_argument("--ckpt", type=str, default="Tiny_NeRF/svrnnn3n/checkpoints/epoch=99-step=10600.ckpt", help="Path to the PyTorch Lightning checkpoint")
    parser.add_argument("--data", type=str, default="tiny_nerf_data.npz", help="Path to the Tiny NeRF dataset")
    parser.add_argument("--output", type=str, default="nerf_360_spin.mp4", help="Output path for the rendered video")
    
    # Rendering parameters
    parser.add_argument("--height", type=int, default=100, help="Image height")
    parser.add_argument("--width", type=int, default=100, help="Image width")
    parser.add_argument("--focal", type=float, default=138.8889, help="Camera focal length")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Number of rays per chunk to prevent OOM")
    
    # Camera trajectory parameters
    parser.add_argument("--frames", type=int, default=40, help="Number of frames for the 360-degree video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--radius", type=float, default=4.0, help="Camera distance from the object")
    parser.add_argument("--elevation", type=float, default=-30.0, help="Camera elevation angle (phi)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize Hydra Configuration
    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = MainConfig(**raw_config)

    # 2. Load the trained NeRF model
    model = NeRF_MLP()
    
    # Load the PyTorch Lightning checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    
    # Remove the 'model.' prefix appended by PyTorch Lightning
    clean_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('model.'):
            clean_state_dict[k.replace('model.', '')] = v
            
    # Inject the cleaned weights into the model
    model.load_state_dict(clean_state_dict)
    
    # Set to evaluation mode and move to GPU
    model.to(device)
    model.eval()
    
    print("✅ Model weights loaded successfully. Ready for rendering.")

    # Optional: Render a single test frame and display it
    tinynerf = TinyNerfDataset(args.data)
    image, test_pose, focal_test = tinynerf.get(100)
    
    img_rendered = render_image(model, H=args.height, W=args.width, focal=args.focal, 
                                cfg=main_cfg, pose=test_pose, device=device, chunk_size=args.chunk_size)

    plt.imshow(img_rendered) # Removed manual * 255 since render_image already returns uint8
    plt.title("NeRF Rendered Test Image")
    plt.show()

    # 3. Render the 360-degree trajectory
    frames = []
    angles = np.linspace(0., 360., args.frames, endpoint=False)
    
    print(f"Starting 360-degree rendering ({args.frames} frames)...")
    for th in angles:
        # Generate spherical camera pose
        pose = pose_spherical(th, args.elevation, args.radius)
        
        # Render the image from the generated pose
        img = render_image(model, H=args.height, W=args.width, focal=args.focal, 
                           pose=pose, cfg=main_cfg, device=device, chunk_size=args.chunk_size)
        
        frames.append(img)
        print(f"📸 Rendered frame at azimuth {th:.1f}°")

    # 4. Export the sequence as an MP4 video
    imageio.mimwrite(args.output, frames, fps=args.fps, quality=8)
    print(f"✅ Video successfully saved to [{args.output}]")