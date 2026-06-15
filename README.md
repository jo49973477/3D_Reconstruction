# 🚀 3D Reconstruction Pipeline: NeRF & 3D Gaussian Splatting

A comprehensive pipeline for training, rendering, and visualizing 3D scenes using Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). This repository provides an end-to-end workflow from model training to mesh extraction, point cloud conversion, and 360-degree video rendering.

## ✨ Features

- **NeRF Training**: Train Neural Radiance Fields using PyTorch Lightning with WandB logging.
- **3DGS Training**: Train 3D Gaussian Splatting models with automatic SfM feature extraction via `pycolmap`.
- **Mesh Extraction**: Extract 3D geometry (PLY meshes) from trained NeRF models using the Marching Cubes algorithm.
- **Point Cloud Conversion**: Convert 3DGS `.pt` checkpoints into universal `.ply` format for compatibility with standard 3D viewers.
- **Novel View Synthesis**: Render 360-degree spin videos from trained NeRF checkpoints.
- **3D Visualization**: Interactive Open3D-based viewer for `.ply` point clouds and meshes.

---

## 🛠️ Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for extremely fast Python package and resolution management.

1. **Install `uv`** (if you haven't already):
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

2. **Clone the repository and sync dependencies**:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
uv sync

```


*Note: `uv sync` will automatically create a virtual environment and install all necessary dependencies (PyTorch, Hydra, gsplat, PyTorch Lightning, Open3D, etc.) specified in the project configuration.*

---

## 📖 Usage Guide

The project is highly configurable via CLI arguments and `hydra` configuration files.

### 1. Training Models

**Train NeRF (PyTorch Lightning)**

```bash
# Uses Hydra for configuration management (conf/tless_config.yaml)
python train_nerf.py

```

**Train 3D Gaussian Splatting**

```bash
# Automatically runs COLMAP SfM if sparse features are not found
python train_gsplat.py

```

### 2. Geometry Extraction & Conversion

**Extract 3D Mesh from NeRF**
Converts NeRF density outputs into a 3D mesh using Marching Cubes.

```bash
python nerf_to_ply.py --checkpoint checkpoints/your_nerf_model.ckpt --output extracted_mesh.ply --grid-res 256

```

**Convert 3DGS Checkpoint to Universal PLY**
Converts proprietary 3DGS `.pt` weights into a standard RGB point cloud.

```bash
python 3dgs_to_ply.py --input 3dgs_checkpoint/latest.pt --output universal_3dgs.ply

```

### 3. Rendering & Visualization

**Render 360° Spin Video (NeRF)**
Generates an `.mp4` video interpolating spherical camera poses.

```bash
python render_frame.py --ckpt checkpoints/your_nerf_model.ckpt --frames 60 --fps 30 --output spin_video.mp4

```

**View 3D Point Clouds / Meshes**
Interactive 3D viewer powered by Open3D.

```bash
# Customize point size and background color
python show_ply.py --input universal_3dgs.ply --point-size 5.0 --bg-color 25 25 25

```

---

## 📁 Repository Structure

* `train_nerf.py`: Main entry point for NeRF training.
* `train_gsplat.py`: Main entry point for 3DGS training.
* `render_frame.py`: Novel view synthesis and video rendering script.
* `nerf_to_ply.py`: Marching cubes mesh extraction from NeRF.
* `3dgs_to_ply.py`: 3DGS tensor to PLY conversion tool.
* `show_ply.py`: Universal Open3D PLY viewer.
* `network/`: Contains MLP architectures and Positional Encoders.
* `dataset/`: Dataloaders for TinyNeRF and T-LESS datasets.
* `conf/` & `conf_gsplat/`: Hydra YAML configuration directories.

## 📊 Logging

This project integrates seamlessly with **Weights & Biases (WandB)**. Make sure you are logged in to track your experiments, loss curves, and rendered step images:

```bash
wandb login

```
