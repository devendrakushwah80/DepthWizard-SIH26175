"""
DepthWizard (SIH26175) — Slope Analysis & Heatmap Generation
Player 3: 3D Reconstruction & Visualization Lead

Computes physical terrain slope angle in degrees considering horizontal GSD
and creates slope color heatmaps for disaster management and site assessment.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def compute_slope_degrees(height_raster: np.ndarray, gsd: float = 0.5) -> np.ndarray:
    """
    Calculates surface slope in degrees using central finite differences (Horn's method).
    
    height_raster: (H, W) metric elevation array in metres
    gsd: Ground Sample Distance in metres (horizontal pixel spacing)
    
    Returns:
      (H, W) float32 array of slope in degrees [0..90]
    """
    H, W = height_raster.shape
    
    # Compute horizontal spatial gradients: dz/dx and dz/dy in m/m
    dz_dy, dz_dx = np.gradient(height_raster.astype(np.float32), gsd, gsd)
    
    # Slope in radians: arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    
    # Convert to degrees
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)
    return slope_deg

def generate_slope_heatmap_texture(slope_deg: np.ndarray, max_slope: float = 60.0, cmap_name: str = "magma") -> Image.Image:
    """
    Renders a colorized slope heatmap image as a PIL Image suitable for 3D texture mapping.
    
    0° (flat, safe): Dark/low color
    30°-60°+ (steep cliff/wall): Bright/high risk color
    """
    norm_slope = np.clip(slope_deg / max_slope, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(norm_slope)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)

def generate_height_heatmap_texture(height_raster: np.ndarray, vmin: float = None, vmax: float = None, cmap_name: str = "turbo") -> Image.Image:
    """
    Renders a colorized metric height heatmap image as a PIL Image suitable for 3D texture mapping.
    """
    if vmin is None:
        vmin = max(0.0, float(np.percentile(height_raster, 1)))
    if vmax is None:
        vmax = max(15.0, float(np.percentile(height_raster, 99)))
        
    norm_h = np.clip((height_raster - vmin) / max(1e-4, vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colored = (cmap(norm_h)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)

def generate_semantic_overlay_texture(rgb: np.ndarray, semantic_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Renders a semi-transparent semantic class color overlay on the optical RGB image.
    
    GAMUS Class Palette:
      0: Others (Gray)
      1: Ground (Khaki)
      2: Low Veg (Light Green)
      3: Buildings (Crimson Red)
      4: Water (Cyan/Blue)
      5: Road (Orange/Yellow)
      6: Tree (Dark Forest Green)
    """
    palette = {
        0: [128, 128, 128], # Others
        1: [210, 180, 140], # Ground
        2: [144, 238, 144], # Low veg
        3: [220, 20, 60],   # Buildings (Crimson)
        4: [30, 144, 255],  # Water (DodgerBlue)
        5: [255, 165, 0],   # Road (Orange)
        6: [34, 139, 34]    # Tree (ForestGreen)
    }
    
    H, W = semantic_mask.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, color in palette.items():
        mask = (semantic_mask == cid)
        overlay[mask] = color
        
    blended = (rgb.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(blended)
