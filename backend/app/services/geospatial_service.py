"""
DepthWizard (SIH26175) — Geospatial & DEM Service
Player 4: Backend & Systems Integration Lead

Handles GeoTIFF metadata parsing (CRS, Affine transform, Bounds, GSD),
base DEM ingestion, and georeferenced raster export (AGL and Absolute DSM).
"""

import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine

class GeospatialService:
    def parse_geotiff(self, file_path: str) -> dict:
        """
        Extracts spatial metadata and RGB array from GeoTIFF file.
        """
        with rasterio.open(file_path) as src:
            crs_str = src.crs.to_string() if src.crs else None
            transform = list(src.transform)
            bounds = {
                'left': float(src.bounds.left),
                'bottom': float(src.bounds.bottom),
                'right': float(src.bounds.right),
                'top': float(src.bounds.top)
            }
            width = src.width
            height = src.height
            
            # Preserve native resolution. Only projected metre CRSs can provide
            # a defensible gsd_m without an external conversion model.
            res_x, res_y = src.res
            has_transform = src.transform is not None and not src.transform.is_identity
            is_georeferenced = src.crs is not None and has_transform
            is_metric_projected = bool(
                is_georeferenced
                and src.crs.is_projected
                and (src.crs.linear_units or "").lower() in {"metre", "meter"}
            )
            gsd_m = float((abs(res_x) + abs(res_y)) / 2.0) if is_metric_projected else None
            
            # Read first 3 bands as RGB
            count = src.count
            if count >= 3:
                rgb_arr = src.read([1, 2, 3])
                rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
            elif count == 1:
                single = src.read(1)
                rgb_arr = np.stack([single, single, single], axis=-1)
            else:
                rgb_arr = src.read()
                rgb_arr = np.transpose(rgb_arr, (1, 2, 0))[:, :, :3]

            # Normalize to uint8 if needed
            if rgb_arr.dtype != np.uint8:
                if rgb_arr.max() <= 1.0:
                    rgb_uint8 = (np.clip(rgb_arr, 0, 1) * 255).astype(np.uint8)
                else:
                    rgb_uint8 = np.clip(rgb_arr, 0, 255).astype(np.uint8)
            else:
                rgb_uint8 = rgb_arr

        return {
            'is_georeferenced': is_georeferenced,
            'crs': crs_str,
            'transform': transform,
            'bounds': bounds,
            'width': width,
            'height': height,
            'gsd_m': gsd_m,
            'gsd_source': 'geotiff' if gsd_m is not None else 'unavailable',
            'resolution': [float(abs(res_x)), float(abs(res_y))],
            'horizontal_units': 'metres' if gsd_m is not None else None,
            'rgb_uint8': rgb_uint8
        }

    def align_and_compute_dsm(self,
                              predicted_agl: np.ndarray,
                              base_dem_path: str,
                              spatial_meta: dict) -> dict:
        """
        Aligns base DEM to the optical grid and calculates: Absolute DSM = Base DEM + Predicted AGL
        """
        if not base_dem_path or not os.path.exists(base_dem_path):
            return {
                'absolute_dsm_available': False,
                'base_dem': None,
                'absolute_dsm': None
            }

        H, W = predicted_agl.shape
        if not spatial_meta.get('is_georeferenced'):
            raise ValueError("DEM alignment validation failed: optical input is not georeferenced")
        target_crs = spatial_meta.get('crs')
        target_transform = Affine(*spatial_meta.get('transform')[:6]) if spatial_meta.get('transform') else None
        if target_crs is None or target_transform is None:
            raise ValueError("DEM alignment validation failed: optical CRS/transform is missing")

        with rasterio.open(base_dem_path) as dem_src:
            failures = []
            if dem_src.count != 1:
                failures.append(f"expected one DEM band, got {dem_src.count}")
            if dem_src.shape != (H, W):
                failures.append(f"shape {dem_src.shape} != optical {(H, W)}")
            dem_crs = dem_src.crs.to_string() if dem_src.crs else None
            if dem_crs != target_crs:
                failures.append(f"CRS {dem_crs!r} != optical {target_crs!r}")
            if not dem_src.transform.almost_equals(target_transform, precision=1e-9):
                failures.append("affine transform/origin differs from optical grid")
            target_res = (abs(target_transform.a), abs(target_transform.e))
            dem_res = (abs(dem_src.transform.a), abs(dem_src.transform.e))
            if not np.allclose(dem_res, target_res, rtol=0.0, atol=1e-9):
                failures.append(f"resolution {dem_res} != optical {target_res}")
            if failures:
                raise ValueError("DEM alignment validation failed: " + "; ".join(failures))

            aligned_dem = dem_src.read(1).astype(np.float32)
            invalid = ~np.isfinite(aligned_dem)
            if dem_src.nodata is not None:
                invalid |= np.isclose(aligned_dem, float(dem_src.nodata))
            if invalid.any():
                raise ValueError(
                    f"DEM alignment validation failed: {int(invalid.sum())} invalid/nodata pixels"
                )

        absolute_dsm = aligned_dem + predicted_agl

        return {
            'absolute_dsm_available': True,
            'base_dem': aligned_dem,
            'absolute_dsm': absolute_dsm
        }

    def export_geotiff(self,
                       raster_data: np.ndarray,
                       output_path: str,
                       spatial_meta: dict):
        """
        Writes a float32 single-band GeoTIFF with CRS and Affine transform.
        """
        H, W = raster_data.shape
        crs = spatial_meta.get('crs')
        transform_list = spatial_meta.get('transform')
        transform = Affine(*transform_list[:6]) if transform_list else Affine.identity()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=1,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(raster_data.astype(np.float32), 1)

geospatial_service = GeospatialService()
