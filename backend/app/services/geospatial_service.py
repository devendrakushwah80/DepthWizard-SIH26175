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
            
            # Preserve native resolution.
            # Projected metre CRSs provide gsd_m directly from linear units.
            # Geographic CRSs (e.g. EPSG:4326) calculate geodesic ground scale at scene latitude
            # to ensure degree units are never conflated with metres.
            res_x, res_y = src.res
            has_transform = src.transform is not None and not src.transform.is_identity
            is_georeferenced = src.crs is not None and has_transform
            is_metric_projected = bool(
                is_georeferenced
                and src.crs.is_projected
                and (src.crs.linear_units or "").lower() in {"metre", "meter"}
            )
            is_geographic = bool(
                is_georeferenced
                and src.crs.is_geographic
            )

            if is_metric_projected:
                gsd_m = float((abs(res_x) + abs(res_y)) / 2.0)
                gsd_source = 'geotiff'
            elif is_geographic:
                center_lat = (float(bounds['top']) + float(bounds['bottom'])) / 2.0
                center_lat_rad = np.deg2rad(center_lat)
                # WGS84 geodesic conversion from angular degrees to metric ground distance
                m_per_deg_lat = (
                    111132.954
                    - 559.822 * np.cos(2.0 * center_lat_rad)
                    + 1.175 * np.cos(4.0 * center_lat_rad)
                )
                m_per_deg_lon = (
                    (np.pi / 180.0)
                    * 6378137.0
                    * np.cos(center_lat_rad)
                    / np.sqrt(max(1e-8, 1.0 - 0.00669437999014 * (np.sin(center_lat_rad) ** 2)))
                )
                gsd_x_m = float(abs(res_x) * m_per_deg_lon)
                gsd_y_m = float(abs(res_y) * m_per_deg_lat)
                gsd_m = float((gsd_x_m + gsd_y_m) / 2.0)
                gsd_source = 'geodesic_wgs84'
            else:
                gsd_m = None
                gsd_source = 'unavailable'
            
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
            'gsd_source': gsd_source,
            'resolution': [float(abs(res_x)), float(abs(res_y))],
            'horizontal_units': 'metres' if gsd_m is not None else None,
            'rgb_uint8': rgb_uint8
        }

    def align_and_compute_dsm(self,
                              predicted_agl: np.ndarray,
                              base_dem_path: str,
                              spatial_meta: dict) -> dict:
        """
        Ingests and aligns base DEM (including coarse DEMs like SRTM 30m) to the optical grid.
        Calculates: Absolute DSM = Base DEM + Predicted AGL.
        Uses rasterio.warp.reproject with continuous bilinear resampling.
        """
        if not base_dem_path or not os.path.exists(base_dem_path):
            return {
                'absolute_dsm_available': False,
                'base_dem': None,
                'absolute_dsm': None,
                'dem_provenance': None
            }

        H, W = predicted_agl.shape
        if not spatial_meta.get('is_georeferenced'):
            raise ValueError("DEM alignment validation failed: optical input is not georeferenced")
        target_crs_str = spatial_meta.get('crs')
        target_transform_list = spatial_meta.get('transform')
        if not target_crs_str or not target_transform_list:
            raise ValueError("DEM alignment validation failed: optical CRS/transform is missing")

        try:
            target_crs = rasterio.crs.CRS.from_string(target_crs_str)
        except Exception as err:
            raise ValueError(f"DEM alignment validation failed: invalid optical CRS '{target_crs_str}': {err}")

        target_transform = Affine(*target_transform_list[:6])

        # Compute optical bounding box in target CRS
        opt_bounds = spatial_meta.get('bounds')
        if opt_bounds:
            opt_left = float(opt_bounds['left'])
            opt_bottom = float(opt_bounds['bottom'])
            opt_right = float(opt_bounds['right'])
            opt_top = float(opt_bounds['top'])
        else:
            opt_left = target_transform.c
            opt_top = target_transform.f
            opt_right = opt_left + target_transform.a * W
            opt_bottom = opt_top + target_transform.e * H
            if opt_bottom > opt_top:
                opt_bottom, opt_top = opt_top, opt_bottom
            if opt_left > opt_right:
                opt_left, opt_right = opt_right, opt_left

        from rasterio.warp import reproject, Resampling, transform_bounds

        with rasterio.open(base_dem_path) as dem_src:
            if dem_src.count < 1:
                raise ValueError("DEM alignment validation failed: DEM has no raster bands")
            if not dem_src.crs:
                raise ValueError("DEM alignment validation failed: DEM CRS is missing where safe alignment cannot be established")

            if dem_src.transform is None or dem_src.transform.is_identity:
                raise ValueError("DEM alignment validation failed: DEM affine transform is invalid or identity")

            dem_bounds = dem_src.bounds
            if not (np.isfinite(dem_bounds.left) and np.isfinite(dem_bounds.bottom) and
                    np.isfinite(dem_bounds.right) and np.isfinite(dem_bounds.top)):
                raise ValueError("DEM alignment validation failed: DEM bounds are not finite")

            # Check spatial overlap by transforming DEM bounds into target optical CRS
            try:
                dem_bounds_in_target = transform_bounds(
                    dem_src.crs,
                    target_crs,
                    dem_bounds.left,
                    dem_bounds.bottom,
                    dem_bounds.right,
                    dem_bounds.top
                )
            except Exception as err:
                raise ValueError(f"DEM alignment validation failed: unable to transform bounds between CRSs: {err}")

            overlap_left = max(opt_left, dem_bounds_in_target[0])
            overlap_bottom = max(opt_bottom, dem_bounds_in_target[1])
            overlap_right = min(opt_right, dem_bounds_in_target[2])
            overlap_top = min(opt_top, dem_bounds_in_target[3])

            if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
                raise ValueError(
                    f"DEM alignment validation failed: DEM does not spatially overlap the optical image "
                    f"(optical: [{opt_left:.1f}, {opt_bottom:.1f}, {opt_right:.1f}, {opt_top:.1f}], "
                    f"dem_in_target: [{dem_bounds_in_target[0]:.1f}, {dem_bounds_in_target[1]:.1f}, "
                    f"{dem_bounds_in_target[2]:.1f}, {dem_bounds_in_target[3]:.1f}])"
                )

            # Check whether DEM is already identical in grid, CRS, and shape
            exact_grid_match = (
                dem_src.crs == target_crs and
                dem_src.shape == (H, W) and
                dem_src.transform.almost_equals(target_transform, precision=1e-5)
            )

            aligned_dem = np.full((H, W), np.nan, dtype=np.float32)

            if exact_grid_match:
                raw_dem = dem_src.read(1).astype(np.float32)
                if dem_src.nodata is not None:
                    raw_dem[np.isclose(raw_dem, float(dem_src.nodata))] = np.nan
                aligned_dem = raw_dem
            else:
                # Reproject and resample onto optical grid using continuous bilinear interpolation
                reproject(
                    source=rasterio.band(dem_src, 1),
                    destination=aligned_dem,
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    src_nodata=dem_src.nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear
                )

            # Check valid coverage and nodata
            valid_mask = np.isfinite(aligned_dem)
            if dem_src.nodata is not None:
                valid_mask &= ~np.isclose(aligned_dem, float(dem_src.nodata))

            total_pixels = H * W
            valid_pixels = int(np.count_nonzero(valid_mask))
            coverage_pct = float((valid_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0
            nodata_pixels = total_pixels - valid_pixels
            nodata_pct = float((nodata_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0

            if valid_pixels == 0 or coverage_pct < 1.0:
                raise ValueError(
                    f"DEM alignment validation failed: unusable DEM coverage ({coverage_pct:.1f}% valid pixels; required >= 1.0%)"
                )

            # Record full DEM provenance
            res_x = float(abs(dem_src.res[0])) if dem_src.res else 0.0
            res_y = float(abs(dem_src.res[1])) if dem_src.res else 0.0
            target_res_x = float(abs(target_transform.a))
            target_res_y = float(abs(target_transform.e))

            dem_provenance = {
                'dem_filename': os.path.basename(base_dem_path),
                'original_dem_crs': dem_src.crs.to_string() if dem_src.crs else None,
                'original_dem_resolution': [res_x, res_y],
                'original_dem_bounds': {
                    'left': float(dem_bounds.left),
                    'bottom': float(dem_bounds.bottom),
                    'right': float(dem_bounds.right),
                    'top': float(dem_bounds.top)
                },
                'resampling_method': 'identity' if exact_grid_match else 'bilinear',
                'aligned_dem_resolution': [target_res_x, target_res_y],
                'coverage_percentage': round(coverage_pct, 2),
                'nodata_pixels': nodata_pixels,
                'nodata_percentage': round(nodata_pct, 2),
                'vertical_datum': 'Vertical datum not verified'
            }

        # Calculate Absolute DSM = Base DEM + Predicted AGL
        absolute_dsm = np.full((H, W), np.nan, dtype=np.float32)
        absolute_dsm[valid_mask] = aligned_dem[valid_mask] + predicted_agl[valid_mask]

        return {
            'absolute_dsm_available': True,
            'base_dem': aligned_dem,
            'absolute_dsm': absolute_dsm,
            'valid_mask': valid_mask,
            'coverage_pct': coverage_pct,
            'dem_provenance': dem_provenance
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
