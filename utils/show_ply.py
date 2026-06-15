import argparse
import open3d as o3d
import numpy as np
import os

def view_ply_open3d(args):
    """
    Loads and visualizes a point cloud from a PLY file using Open3D.
    """
    if not os.path.exists(args.input):
        print(f"Error: The file '{args.input}' does not exist.")
        return

    # 1. Load the PLY file
    print(f"Loading point cloud from {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    
    if not pcd.has_points():
        print("Error: The loaded point cloud contains no points.")
        return

    print(f"Successfully loaded. Number of points: {len(pcd.points)}")
    
    # 2. Initialize the visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Point Cloud Viewer", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # 3. Adjust rendering options
    render_option = vis.get_render_option()
    render_option.point_size = args.point_size 
    
    # Set background color (normalize 0-255 RGB values to 0.0-1.0 range)
    bg_color = np.array(args.bg_color) / 255.0
    render_option.background_color = bg_color 
    
    # 4. Run the visualization loop
    print("Starting visualization. Close the window to terminate the program.")
    vis.run()
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3D point cloud (.ply) using Open3D")
    
    parser.add_argument("--input", type=str, default="universal_output.ply", 
                        help="Path to the input PLY file")
    
    parser.add_argument("--point-size", type=float, default=5.0, 
                        help="Rendered size of the points. Increase this value if the points are too small to see.")
    
    parser.add_argument("--bg-color", type=int, nargs=3, default=[25, 25, 25], 
                        help="Background color in RGB format (0-255). Default is [25, 25, 25] (dark gray).")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    view_ply_open3d(args)