"""
DepthWizard (SIH26175) — Raster to 3D Triangular Mesh Generator
Player 3: 3D Reconstruction & Visualization Lead

Converts continuous 2D metric AGL rasters into watertight 3D textured triangular meshes
with configurable downsampling, normal computation, UV texture mapping, and boundary side skirts.
"""

import numpy as np
import scipy.ndimage
from PIL import Image
from src.geometry.scene import Scene3D

def downsample_raster(raster: np.ndarray, target_size: tuple, method: str = "bilinear") -> np.ndarray:
    """
    Downsamples a 2D height raster to target (out_h, out_w).
    Methods:
      - 'bilinear': Standard bilinear resampling via scipy zoom
      - 'area': Box-area local mean
      - 'max_aware': Preserves localized structural building peaks
    """
    in_h, in_w = raster.shape
    out_h, out_w = target_size
    
    if (in_h, in_w) == (out_h, out_w):
        return raster.copy()
        
    zoom_y = out_h / in_h
    zoom_x = out_w / in_w

    if method == "bilinear":
        return scipy.ndimage.zoom(raster, (zoom_y, zoom_x), order=1, mode='nearest')
    elif method == "area":
        # Anti-aliased full-extent mean approximation that also supports
        # non-divisible and upsampled arbitrary input sizes.
        filter_y = max(1, int(np.ceil(in_h / out_h)))
        filter_x = max(1, int(np.ceil(in_w / out_w)))
        smoothed = scipy.ndimage.uniform_filter(
            raster.astype(np.float32), size=(filter_y, filter_x), mode="nearest"
        )
        return scipy.ndimage.zoom(smoothed, (zoom_y, zoom_x), order=1, mode="nearest")
    elif method == "max_aware":
        # Full-extent peak-aware resampling; no bottom/right pixels are dropped
        # when dimensions are not integer multiples of the mesh resolution.
        filter_y = max(1, int(np.ceil(in_h / out_h)))
        filter_x = max(1, int(np.ceil(in_w / out_w)))
        source = raster.astype(np.float32)
        mean_source = scipy.ndimage.uniform_filter(
            source, size=(filter_y, filter_x), mode="nearest"
        )
        max_source = scipy.ndimage.maximum_filter(
            source, size=(filter_y, filter_x), mode="nearest"
        )
        mean_val = scipy.ndimage.zoom(mean_source, (zoom_y, zoom_x), order=1, mode="nearest")
        max_val = scipy.ndimage.zoom(max_source, (zoom_y, zoom_x), order=1, mode="nearest")
        # Where relief delta is significant, favor peak preservation
        delta = max_val - mean_val
        alpha = np.clip(delta / 5.0, 0.0, 0.6) # max 60% bias towards peak
        return (1.0 - alpha) * mean_val + alpha * max_val
    else:
        raise ValueError(f"Unknown downsample method: {method}")

class Mesh3D:
    def __init__(self,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 normals: np.ndarray,
                 uvs: np.ndarray,
                 texture_image: Image.Image = None,
                 metadata: dict = None):
        self.vertices = vertices.astype(np.float32)  # (N, 3) [X, Y, Z]
        self.faces = faces.astype(np.int32)         # (M, 3) [idx0, idx1, idx2]
        self.normals = normals.astype(np.float32)   # (N, 3)
        self.uvs = uvs.astype(np.float32)           # (N, 2) [u, v]
        self.texture_image = texture_image
        self.metadata = metadata or {}

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.faces)

def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes area-weighted vertex normals for smooth shading."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Face normals (unnormalized cross product is proportional to triangle area)
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # Accumulate onto vertices
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
        
    # Normalize
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return normals / lens

def build_terrain_mesh(scene: Scene3D,
                       mesh_resolution: int = 384,
                       downsample_method: str = "max_aware",
                       vertical_exaggeration: float = 1.0,
                       add_side_skirts: bool = True,
                       base_depth_m: float = 2.0) -> Mesh3D:
    """
    Generates a structured triangulated terrain mesh from a Scene3D.
    
    mesh_resolution: Grid dimension (e.g. 256, 384, 512)
    downsample_method: 'max_aware', 'bilinear', or 'area'
    vertical_exaggeration: 1.0 for physical measurement mode; 1.5/2.0 for visual mode
    add_side_skirts: Generates side walls down to base plane for a solid watertight block
    base_depth_m: Depth below minimum elevation for the base skirt
    """
    N = int(mesh_resolution)
    
    # 1. Downsample height raster
    sampled_h = downsample_raster(scene.surface_height, (N, N), method=downsample_method)
    
    # 2. Metric World Coordinates
    # World width & height in metres
    phys_w = scene.W * scene.xy_scale
    phys_h = scene.H * scene.xy_scale
    
    # Create coordinate grid
    xs = np.linspace(-phys_w / 2.0, phys_w / 2.0, N, dtype=np.float32)
    # Z: row 0 is North (+Z), row N-1 is South (-Z)
    zs = np.linspace(phys_h / 2.0, -phys_h / 2.0, N, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    
    # Y: Elevation in metres (with user-specified vertical exaggeration)
    grid_y = sampled_h * float(vertical_exaggeration)
    
    # Flatten surface vertices: (N*N, 3)
    surf_verts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    
    # UV Coordinates: u in [0, 1] left->right, v in [0, 1] bottom->top
    # Image row 0 corresponds to v=1 (top), row N-1 to v=0 (bottom)
    us = np.linspace(0.0, 1.0, N, dtype=np.float32)
    vs = np.linspace(1.0, 0.0, N, dtype=np.float32) # top is 1.0
    grid_u, grid_v = np.meshgrid(us, vs)
    surf_uvs = np.column_stack([grid_u.ravel(), grid_v.ravel()])
    
    # 3. Create Grid Triangle Faces
    # For each quad (i, j): two triangles
    i_idx, j_idx = np.meshgrid(np.arange(N - 1), np.arange(N - 1), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    
    v00 = i_idx * N + j_idx
    v01 = i_idx * N + (j_idx + 1)
    v10 = (i_idx + 1) * N + j_idx
    v11 = (i_idx + 1) * N + (j_idx + 1)
    
    # Counter-clockwise winding: [v00, v10, v01] and [v01, v10, v11]
    t1 = np.column_stack([v00, v10, v01])
    t2 = np.column_stack([v01, v10, v11])
    surf_faces = np.vstack([t1, t2])
    
    all_verts = [surf_verts]
    all_uvs = [surf_uvs]
    all_faces = [surf_faces]
    
    # 4. Optional Watertight Side Skirts & Base
    if add_side_skirts:
        base_y = float(np.min(grid_y) - base_depth_m)
        curr_vert_offset = len(surf_verts)
        
        # 4 perimeter edges (top: i=0, bottom: i=N-1, left: j=0, right: j=N-1)
        top_indices = np.arange(N)                    # row 0 (left to right)
        right_indices = np.arange(N - 1, N * N, N)     # col N-1 (top to bottom)
        bottom_indices = np.arange(N * N - 1, N * (N - 1) - 1, -1) # row N-1 (right to left)
        left_indices = np.arange(N * (N - 1), -1, -N)  # col 0 (bottom to top)
        
        # Combine into continuous perimeter loop of N*4 vertices
        perimeter_indices = np.concatenate([top_indices, right_indices[1:], bottom_indices[1:], left_indices[1:-1]])
        P = len(perimeter_indices)
        
        # Create bottom skirt vertices directly under perimeter
        skirt_bottom_verts = surf_verts[perimeter_indices].copy()
        skirt_bottom_verts[:, 1] = base_y
        
        # UVs for skirts (clamp to boundary pixel edge)
        skirt_bottom_uvs = surf_uvs[perimeter_indices].copy()
        
        all_verts.append(skirt_bottom_verts)
        all_uvs.append(skirt_bottom_uvs)
        
        # Skirt quad faces
        skirt_faces = []
        for p in range(P):
            p_next = (p + 1) % P
            
            top_curr = perimeter_indices[p]
            top_next = perimeter_indices[p_next]
            bot_curr = curr_vert_offset + p
            bot_next = curr_vert_offset + p_next
            
            # Quad faces: [top_curr, bot_curr, top_next] and [top_next, bot_curr, bot_next]
            skirt_faces.append([top_curr, bot_curr, top_next])
            skirt_faces.append([top_next, bot_curr, bot_next])
            
        all_faces.append(np.array(skirt_faces, dtype=np.int32))

    # Combine all mesh components
    final_vertices = np.vstack(all_verts)
    final_uvs = np.vstack(all_uvs)
    final_faces = np.vstack(all_faces)
    
    # Compute smooth vertex normals
    final_normals = compute_vertex_normals(final_vertices, final_faces)
    
    # Texture Image
    tex_pil = Image.fromarray(scene.rgb)
    
    metadata = {
        'scene_id': scene.scene_id,
        'mesh_resolution': N,
        'vertex_count': len(final_vertices),
        'triangle_count': len(final_faces),
        'vertical_exaggeration': vertical_exaggeration,
        'downsample_method': downsample_method,
        'has_side_skirts': add_side_skirts,
        'physical_width_m': phys_w,
        'physical_height_m': phys_h,
        'horizontal_units': scene.horizontal_units,
        'height_surface': scene.surface_type,
        'min_elevation_m': float(np.min(sampled_h)),
        'max_elevation_m': float(np.max(sampled_h)),
        'mean_elevation_m': float(np.mean(sampled_h))
    }
    
    return Mesh3D(final_vertices, final_faces, final_normals, final_uvs, tex_pil, metadata)
