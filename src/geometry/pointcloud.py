"""
DepthWizard (SIH26175) — Dense 3D Point Cloud Generator & Exporter
Player 3: 3D Reconstruction & Visualization Lead

Generates and exports calibrated 3D colored point clouds in standard PLY format
with per-point spatial coordinates (X, Y, Z), optical RGB colors, and semantic attributes.
"""

import os
import numpy as np
from src.geometry.scene import Scene3D

def export_point_cloud_ply(scene: Scene3D,
                           output_path: str,
                           sampling_step: int = 1,
                           binary: bool = True,
                           vertical_exaggeration: float = 1.0) -> dict:
    """
    Exports a dense 3D point cloud from Scene3D to a standard .ply file.
    
    sampling_step: 1 = full 1024x1024 (1M pts), 2 = 512x512 (262k pts), 4 = 256x256 (65k pts)
    binary: True for compact high-speed binary PLY, False for human-readable ASCII
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    step = int(sampling_step)
    sub_h = scene.surface_height[::step, ::step]
    sub_agl = scene.predicted_agl[::step, ::step]
    sub_rgb = scene.rgb[::step, ::step]
    sub_sem = scene.semantic[::step, ::step]
    sub_mask = scene.valid_mask[::step, ::step]
    
    N_y, N_x = sub_h.shape
    phys_w = scene.W * scene.xy_scale
    phys_h = scene.H * scene.xy_scale
    
    xs = np.linspace(-phys_w / 2.0, phys_w / 2.0, N_x, dtype=np.float32)
    zs = np.linspace(phys_h / 2.0, -phys_h / 2.0, N_y, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    
    grid_y = sub_h * float(vertical_exaggeration)
    
    # Flatten arrays
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    flat_z = grid_z.ravel()
    flat_r = sub_rgb[:, :, 0].ravel()
    flat_g = sub_rgb[:, :, 1].ravel()
    flat_b = sub_rgb[:, :, 2].ravel()
    flat_cls = sub_sem.ravel()
    flat_agl = sub_agl.ravel()
    flat_valid = sub_mask.ravel()
    
    # Filter valid points
    valid_idx = np.where(flat_valid)[0]
    num_points = len(valid_idx)
    
    x = flat_x[valid_idx]
    y = flat_y[valid_idx]
    z = flat_z[valid_idx]
    r = flat_r[valid_idx]
    g = flat_g[valid_idx]
    b = flat_b[valid_idx]
    cls_id = flat_cls[valid_idx]
    agl = flat_agl[valid_idx]
    
    if binary:
        # Structured binary numpy array
        dtype = [
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('class_id', 'i2'), ('agl_height', '<f4')
        ]
        structured_arr = np.empty(num_points, dtype=dtype)
        structured_arr['x'] = x
        structured_arr['y'] = y
        structured_arr['z'] = z
        structured_arr['red'] = r
        structured_arr['green'] = g
        structured_arr['blue'] = b
        structured_arr['class_id'] = cls_id.astype(np.int16)
        structured_arr['agl_height'] = agl
        
        header = f"""ply
format binary_little_endian 1.0
comment DepthWizard SIH26175 Generated Point Cloud
comment Scene: {scene.scene_id} | Sampling: {step}x
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property short class_id
property float agl_height
end_header
"""
        with open(output_path, 'wb') as f:
            f.write(header.encode('ascii'))
            structured_arr.tofile(f)
    else:
        header = f"""ply
format ascii 1.0
comment DepthWizard SIH26175 Generated Point Cloud
comment Scene: {scene.scene_id}
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property short class_id
property float agl_height
end_header
"""
        with open(output_path, 'w', encoding='ascii') as f:
            f.write(header)
            for i in range(num_points):
                f.write(f"{x[i]:.3f} {y[i]:.3f} {z[i]:.3f} {r[i]} {g[i]} {b[i]} {cls_id[i]} {agl[i]:.2f}\n")
                
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return {
        'output_path': output_path,
        'point_count': num_points,
        'sampling_step': step,
        'file_size_mb': file_size_mb,
        'format': 'binary_ply' if binary else 'ascii_ply'
        ,'horizontal_units': scene.horizontal_units
        ,'height_surface': scene.surface_type
    }
