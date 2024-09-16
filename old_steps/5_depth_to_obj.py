import numpy as np
import cv2
import open3d as o3d

def depth_to_point_cloud(depth_map, mask, camera_intrinsics):
    """
    Convert a depth map to a 3D point cloud.

    Parameters:
    - depth_map: The depth map.
    - mask: The mask defining the region of interest.
    - camera_intrinsics: The intrinsic parameters of the camera.

    Returns:
    - points: Nx3 array of 3D points.
    """
    fx, fy, cx, cy = camera_intrinsics

    height, width = depth_map.shape
    points = []

    for v in range(height):
        for u in range(width):
            if mask[v, u] > 0:
                z = depth_map[v, u] / 255.0  # Scale depth to match actual depth values
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

    points = np.array(points)
    return points

# Example camera intrinsics (you may need to adjust these based on your camera)
camera_intrinsics = [fx, fy, cx, cy]  # Fill these with your camera's intrinsic parameters

# Process the frames and generate point clouds
for i, (depth_map_file, mask_file) in enumerate(zip(depth_map_files, mask_files)):
    depth_map = cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    points = depth_to_point_cloud(depth_map, mask, camera_intrinsics)

    # Save or visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(f'point_cloud_{i:04d}.ply', point_cloud)
    print(f"Point cloud for frame {i} saved.")

import open3d as o3d
# Load and visualize a point cloud
point_cloud = o3d.io.read_point_cloud("point_cloud_0000.ply")
o3d.visualization.draw_geometries([point_cloud])
