"""
DepthWizard (SIH26175) — Standard 3D Scene Representation
Player 3: 3D Reconstruction & Visualization Lead

Defines the core Scene3D intermediate data structure supporting both
georeferenced and non-georeferenced scenes with metric height calibration.
"""

import os
import json
import numpy as np
from PIL import Image

class Scene3D:
    def __init__(self,
                 scene_id: str,
                 rgb: np.ndarray,
                 predicted_agl: np.ndarray,
                 semantic: np.ndarray = None,
                 valid_mask: np.ndarray = None,
                 gsd: float = None,
                 is_georeferenced: bool = False,
                 crs: str = None,
                 transform: list = None,
                 bounds: dict = None,
                 base_dem: np.ndarray = None):
        """
        scene_id: Unique string identifier (e.g., 'NYC_00735')
        rgb: (H, W, 3) uint8 or float32 [0..1]
        predicted_agl: (H, W) float32 metric height Above Ground Level in metres
        semantic: (H, W) int64 GAMUS class IDs (0..6)
        valid_mask: (H, W) bool
        gsd: Ground Sample Distance in metres/pixel, or None when unknown
        is_georeferenced: bool
        crs: Coordinate Reference System string (e.g. 'EPSG:32618')
        transform: Affine transform matrix elements
        bounds: Geographic bounding box {left, right, top, bottom}
        base_dem: (H, W) float32 aligned terrain DEM elevation in metres
        """
        self.scene_id = scene_id
        
        # Ensure RGB is uint8 (H, W, 3)
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                self.rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            else:
                self.rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            self.rgb = rgb
            
        self.H, self.W, _ = self.rgb.shape
        self.predicted_agl = predicted_agl.astype(np.float32)
        
        if semantic is not None:
            self.semantic = semantic.astype(np.int64)
        else:
            self.semantic = np.zeros((self.H, self.W), dtype=np.int64)
            
        if valid_mask is not None:
            self.valid_mask = valid_mask.astype(bool)
        else:
            self.valid_mask = (self.predicted_agl >= 0) & np.isfinite(self.predicted_agl)
            
        self.gsd = float(gsd) if gsd is not None else None
        self.xy_scale = self.gsd if self.gsd is not None else 1.0
        self.horizontal_units = "metres" if self.gsd is not None else "pixels"
        self.is_georeferenced = is_georeferenced
        self.crs = crs
        self.transform = transform
        self.bounds = bounds
        self.base_dem = base_dem.astype(np.float32) if base_dem is not None else None
        
        # Compute absolute DSM = Base DEM + Predicted AGL
        if self.base_dem is not None:
            self.absolute_dsm = self.base_dem + self.predicted_agl
        else:
            self.absolute_dsm = None

    @property
    def physical_width_m(self) -> float:
        return self.W * self.gsd if self.gsd is not None else None

    @property
    def physical_height_m(self) -> float:
        return self.H * self.gsd if self.gsd is not None else None

    @property
    def surface_height(self) -> np.ndarray:
        """Raster used for 3D geometry: absolute DSM when valid, else local AGL."""
        return self.absolute_dsm if self.absolute_dsm is not None else self.predicted_agl

    @property
    def surface_type(self) -> str:
        return "absolute_dsm" if self.absolute_dsm is not None else "predicted_agl"

    def get_elevation_at_pixel(self, u: int, v: int) -> dict:
        """Queries precise full-resolution height and semantic class at image pixel (u, v)."""
        u = int(np.clip(u, 0, self.W - 1))
        v = int(np.clip(v, 0, self.H - 1))
        
        agl = float(self.predicted_agl[v, u])
        sem = int(self.semantic[v, u])
        valid = bool(self.valid_mask[v, u])
        
        # Metric world coordinates relative to center
        world_x = (u - self.W / 2.0) * self.gsd if self.gsd is not None else None
        world_z = (self.H / 2.0 - v) * self.gsd if self.gsd is not None else None
        
        abs_elev = float(self.absolute_dsm[v, u]) if self.absolute_dsm is not None else None
        
        return {
            'pixel_u': u,
            'pixel_v': v,
            'world_x_m': world_x,
            'world_z_m': world_z,
            'agl_height_m': agl,
            'absolute_elevation_m': abs_elev,
            'semantic_class_id': sem,
            'is_valid': valid,
            'is_georeferenced': self.is_georeferenced
        }

    def to_metadata_dict(self) -> dict:
        valid_agl = self.predicted_agl[self.valid_mask]
        return {
            'scene_id': self.scene_id,
            'raster_width': self.W,
            'raster_height': self.H,
            'gsd_m_per_px': self.gsd,
            'horizontal_units': self.horizontal_units,
            'physical_width_m': self.physical_width_m,
            'physical_height_m': self.physical_height_m,
            'is_georeferenced': self.is_georeferenced,
            'crs': self.crs,
            'bounds': self.bounds,
            'min_agl_m': float(np.min(valid_agl)) if len(valid_agl) > 0 else 0.0,
            'max_agl_m': float(np.max(valid_agl)) if len(valid_agl) > 0 else 0.0,
            'mean_agl_m': float(np.mean(valid_agl)) if len(valid_agl) > 0 else 0.0,
            'median_agl_m': float(np.median(valid_agl)) if len(valid_agl) > 0 else 0.0,
            'std_agl_m': float(np.std(valid_agl)) if len(valid_agl) > 0 else 0.0
        }
