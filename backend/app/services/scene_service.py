"""Scene product generation and authoritative raster inspection."""

import json
import os
import shutil
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.geometry.scene import Scene3D
from src.geometry.raster_to_mesh import build_terrain_mesh
from src.geometry.gltf_export import export_mesh_glb, export_mesh_obj
from src.geometry.pointcloud import export_point_cloud_ply
from src.geometry.slope import (
    compute_slope_degrees,
    generate_slope_heatmap_texture,
    generate_height_heatmap_texture,
    generate_semantic_overlay_texture,
)
from backend.app.config import settings


GSD_WARNING = "GSD unavailable: horizontal metric distance and slope disabled."
DSM_WARNING = "Absolute DSM unavailable: aligned terrain DEM was not supplied."


class SceneService:
    def __init__(self):
        self.storage_dir = settings.STORAGE_DIR

    def get_scene_dir(self, scene_id: str) -> str:
        return os.path.join(self.storage_dir, scene_id)

    def scene_exists(self, scene_id: str) -> bool:
        return os.path.exists(
            os.path.join(self.get_scene_dir(scene_id), "metadata", "scene.json")
        )

    def load_scene_metadata(self, scene_id: str) -> Optional[dict]:
        meta_path = os.path.join(self.get_scene_dir(scene_id), "metadata", "scene.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _write_metadata(self, scene_id: str, metadata: dict):
        metadata_dir = os.path.join(self.get_scene_dir(scene_id), "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        with open(os.path.join(metadata_dir, "scene.json"), "w", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2)

    @staticmethod
    def _artifact_sizes(scene_dir: str) -> dict:
        mapping = {
            "predicted_agl_npy": os.path.join("rasters", "predicted_agl.npy"),
            "predicted_agl_tif": os.path.join("rasters", "predicted_agl.tif"),
            "relative_surface_npy": os.path.join("rasters", "relative_surface.npy"),
            "relative_surface_png": os.path.join("textures", "relative_surface.png"),
            "base_dem_npy": os.path.join("rasters", "base_dem.npy"),
            "aligned_terrain_dem_tif": os.path.join("rasters", "aligned_terrain_dem.tif"),
            "absolute_dsm_npy": os.path.join("rasters", "absolute_dsm.npy"),
            "absolute_dsm_tif": os.path.join("rasters", "absolute_dsm.tif"),
            "height_visualization_png": os.path.join("textures", "height_heatmap.png"),
            "slope_visualization_png": os.path.join("textures", "slope.png"),
            "rgb_texture_jpg": os.path.join("textures", "rgb.jpg"),
            "mesh_glb": os.path.join("mesh", "scene.glb"),
            "mesh_agl_glb": os.path.join("mesh", "scene_agl.glb"),
            "mesh_relative_glb": os.path.join("mesh", "scene_relative.glb"),
            "mesh_absolute_dsm_glb": os.path.join("mesh", "scene_absolute_dsm.glb"),
            "mesh_obj": os.path.join("mesh", "scene.obj"),
            "pointcloud_ply": os.path.join("pointcloud", "scene.ply"),
        }
        return {
            name: os.path.getsize(os.path.join(scene_dir, relative_path))
            for name, relative_path in mapping.items()
            if os.path.exists(os.path.join(scene_dir, relative_path))
        }

    def create_scene_products(
        self,
        scene_id: str,
        rgb_image: np.ndarray,
        predicted_agl: np.ndarray,
        spatial_meta: dict,
        base_dem: np.ndarray = None,
        relative_surface: np.ndarray = None,
        dem_provenance: dict = None,
        semantic_mask: np.ndarray = None,
        generate_3d: bool = True,
        generate_pointcloud: bool = True,
        mesh_resolution: int = 384,
        vertical_exaggeration: float = 1.0,
        processing_timings: Optional[dict] = None,
        peak_vram_mb: float = 0.0,
    ) -> dict:
        """Create AGL/DSM rasters, visualizations and physical 1x 3D products."""
        if abs(float(vertical_exaggeration) - 1.0) > 1e-9:
            raise ValueError(
                "3D artifacts must remain at 1.0x physical height; apply vertical "
                "exaggeration only in the viewer"
            )
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError("Scene RGB must have shape (height, width, 3)")
        if predicted_agl.shape != rgb_image.shape[:2]:
            raise ValueError("Predicted AGL shape does not match the uploaded RGB grid")
        if not np.isfinite(predicted_agl).all() or np.any(predicted_agl < 0):
            raise ValueError("Predicted AGL contains invalid values")
        if base_dem is not None and base_dem.shape != predicted_agl.shape:
            raise ValueError("Aligned DEM shape does not match predicted AGL")

        t0 = time.perf_counter()
        scene_dir = self.get_scene_dir(scene_id)
        dir_rasters = os.path.join(scene_dir, "rasters")
        dir_mesh = os.path.join(scene_dir, "mesh")
        dir_pointcloud = os.path.join(scene_dir, "pointcloud")
        dir_textures = os.path.join(scene_dir, "textures")
        dir_metadata = os.path.join(scene_dir, "metadata")
        for directory in (dir_rasters, dir_mesh, dir_pointcloud, dir_textures, dir_metadata):
            os.makedirs(directory, exist_ok=True)

        height, width, _ = rgb_image.shape
        gsd_value = spatial_meta.get("gsd_m")
        gsd_m = float(gsd_value) if gsd_value is not None else None
        is_geo = bool(spatial_meta.get("is_georeferenced", False))
        crs = spatial_meta.get("crs")
        bounds = spatial_meta.get("bounds")
        transform = spatial_meta.get("transform")

        np.save(os.path.join(dir_rasters, "predicted_agl.npy"), predicted_agl.astype(np.float32))
        if relative_surface is not None:
            np.save(os.path.join(dir_rasters, "relative_surface.npy"), relative_surface.astype(np.float32))
            generate_height_heatmap_texture(
                relative_surface, vmin=0.0, vmax=1.0, cmap_name="turbo"
            ).save(os.path.join(dir_textures, "relative_surface.png"), "PNG")

        if base_dem is not None:
            np.save(os.path.join(dir_rasters, "base_dem.npy"), base_dem.astype(np.float32))
            np.save(
                os.path.join(dir_rasters, "absolute_dsm.npy"),
                (base_dem + predicted_agl).astype(np.float32),
            )

        slope_deg = None
        if gsd_m is not None:
            slope_deg = compute_slope_degrees(predicted_agl, gsd=gsd_m)
            np.save(os.path.join(dir_rasters, "slope.npy"), slope_deg)

        Image.fromarray(rgb_image.astype(np.uint8)).save(
            os.path.join(dir_textures, "rgb.jpg"), "JPEG", quality=90
        )
        vmax_h = max(20.0, float(np.percentile(predicted_agl, 99)))
        generate_height_heatmap_texture(
            predicted_agl, vmin=0.0, vmax=vmax_h, cmap_name="turbo"
        ).save(os.path.join(dir_textures, "height_heatmap.png"), "PNG")
        if slope_deg is not None:
            generate_slope_heatmap_texture(
                slope_deg, max_slope=60.0, cmap_name="magma"
            ).save(os.path.join(dir_textures, "slope.png"), "PNG")
        if semantic_mask is not None:
            generate_semantic_overlay_texture(
                rgb_image, semantic_mask, alpha=0.5
            ).save(os.path.join(dir_textures, "semantic.png"), "PNG")

        scene_obj = Scene3D(
            scene_id=scene_id,
            rgb=rgb_image,
            predicted_agl=predicted_agl,
            semantic=semantic_mask,
            gsd=gsd_m,
            is_georeferenced=is_geo,
            crs=crs,
            transform=transform,
            bounds=bounds,
            base_dem=base_dem,
        )
        agl_scene_obj = Scene3D(
            scene_id=scene_id,
            rgb=rgb_image,
            predicted_agl=predicted_agl,
            semantic=semantic_mask,
            gsd=gsd_m,
            is_georeferenced=is_geo,
            crs=crs,
            transform=transform,
            bounds=bounds,
            base_dem=None,
        )

        mesh_meta = {}
        mesh_generation_s = 0.0
        glb_generation_s = 0.0
        obj_generation_s = 0.0
        relative_mesh_generation_s = 0.0
        relative_mesh_meta = {}
        absolute_mesh_generation_s = 0.0
        absolute_mesh_meta = {}
        if generate_3d:
            mesh_start = time.perf_counter()
            mesh_3d = build_terrain_mesh(
                agl_scene_obj,
                mesh_resolution=mesh_resolution,
                downsample_method="max_aware",
                vertical_exaggeration=1.0,
                add_side_skirts=True,
            )
            mesh_generation_s = time.perf_counter() - mesh_start

            glb_start = time.perf_counter()
            default_glb_path = os.path.join(dir_mesh, "scene.glb")
            mesh_meta = export_mesh_glb(mesh_3d, default_glb_path)
            shutil.copy2(default_glb_path, os.path.join(dir_mesh, "scene_agl.glb"))
            glb_generation_s = time.perf_counter() - glb_start

            obj_start = time.perf_counter()
            export_mesh_obj(mesh_3d, os.path.join(dir_mesh, "scene.obj"))
            obj_generation_s = time.perf_counter() - obj_start

            if relative_surface is not None:
                rel_mesh_start = time.perf_counter()
                rel_scene_obj = Scene3D(
                    scene_id=scene_id,
                    rgb=rgb_image,
                    predicted_agl=relative_surface * 25.0,
                    semantic=semantic_mask,
                    gsd=gsd_m,
                    is_georeferenced=is_geo,
                    crs=crs,
                    transform=transform,
                    bounds=bounds,
                    base_dem=None,
                )
                rel_mesh = build_terrain_mesh(
                    rel_scene_obj,
                    mesh_resolution=mesh_resolution,
                    downsample_method="max_aware",
                    vertical_exaggeration=1.0,
                    add_side_skirts=True,
                )
                relative_mesh_meta = export_mesh_glb(
                    rel_mesh,
                    os.path.join(dir_mesh, "scene_relative.glb"),
                )
                relative_mesh_generation_s = time.perf_counter() - rel_mesh_start

            if base_dem is not None:
                absolute_mesh_start = time.perf_counter()
                absolute_mesh = build_terrain_mesh(
                    scene_obj,
                    mesh_resolution=mesh_resolution,
                    downsample_method="max_aware",
                    vertical_exaggeration=1.0,
                    add_side_skirts=True,
                )
                absolute_mesh_meta = export_mesh_glb(
                    absolute_mesh,
                    os.path.join(dir_mesh, "scene_absolute_dsm.glb"),
                )
                absolute_mesh_generation_s = time.perf_counter() - absolute_mesh_start

            available_surfaces = ["predicted_agl"]
            if relative_surface is not None:
                available_surfaces.insert(0, "relative_surface")
            if base_dem is not None:
                available_surfaces.append("absolute_dsm")

            mesh_meta["default_surface"] = (
                "relative_surface" if (not is_geo and relative_surface is not None) else "predicted_agl"
            )
            mesh_meta["available_surfaces"] = available_surfaces

        ply_meta = {}
        pointcloud_generation_s = 0.0
        if generate_pointcloud:
            ply_start = time.perf_counter()
            ply_meta = export_point_cloud_ply(
                scene_obj,
                os.path.join(dir_pointcloud, "scene.ply"),
                sampling_step=2,
                binary=True,
                vertical_exaggeration=1.0,
            )
            pointcloud_generation_s = time.perf_counter() - ply_start

        warnings = []
        if gsd_m is None:
            warnings.append(GSD_WARNING)
        if base_dem is None:
            warnings.append(DSM_WARNING)

        products = {
            "predicted_agl_npy": True,
            "predicted_agl_tif": os.path.exists(os.path.join(dir_rasters, "predicted_agl.tif")),
            "relative_surface_npy": relative_surface is not None,
            "relative_surface_png": relative_surface is not None,
            "aligned_terrain_dem_tif": os.path.exists(os.path.join(dir_rasters, "aligned_terrain_dem.tif")),
            "absolute_dsm_tif": os.path.exists(os.path.join(dir_rasters, "absolute_dsm.tif")),
            "mesh_glb": generate_3d,
            "mesh_agl_glb": generate_3d,
            "mesh_relative_glb": generate_3d and relative_surface is not None,
            "mesh_absolute_dsm_glb": generate_3d and base_dem is not None,
            "mesh_obj": generate_3d,
            "pointcloud_ply": generate_pointcloud,
            "texture_rgb": True,
            "texture_height_heatmap": True,
            "texture_relative_surface": relative_surface is not None,
            "texture_semantic": semantic_mask is not None,
            "texture_slope_heatmap": slope_deg is not None,
        }
        timings = dict(processing_timings or {})
        timings.update(
            {
                "mesh_generation_s": round(mesh_generation_s, 6),
                "glb_generation_s": round(glb_generation_s, 6),
                "obj_generation_s": round(obj_generation_s, 6),
                "relative_mesh_generation_s": round(relative_mesh_generation_s, 6),
                "absolute_mesh_generation_s": round(absolute_mesh_generation_s, 6),
                "pointcloud_generation_s": round(pointcloud_generation_s, 6),
            }
        )

        metadata = {
            "scene_id": scene_id,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": {
                "identity": settings.MODEL_NAME,
                "checkpoint_sha256": settings.MODEL_CHECKPOINT_SHA256,
                "output_parameterization": settings.MODEL_OUTPUT_PARAMETERIZATION,
                "dav2_model_id": settings.DAV2_MODEL_ID,
                "dav2_frozen": True,
            },
            "input_info": {
                "format": spatial_meta.get("format", "RGB Image"),
                "width": width,
                "height": height,
                "channels": 3,
                "is_georeferenced": is_geo,
            },
            "spatial_info": {
                "is_georeferenced": is_geo,
                "crs": crs,
                "gsd_m": gsd_m,
                "gsd_source": spatial_meta.get("gsd_source", "unavailable"),
                "horizontal_units": "metres" if gsd_m is not None else "pixels",
                "physical_width_m": scene_obj.physical_width_m,
                "physical_height_m": scene_obj.physical_height_m,
                "bounds": bounds,
                "transform": transform,
                "resolution": spatial_meta.get("resolution"),
            },
            "dem_provenance": dem_provenance,
            "output_semantics": {
                "primary": "predicted_agl_ndsm",
                "vertical_unit": "metres",
                "horizontal_metric_scale_known": gsd_m is not None,
                "absolute_dsm_available": base_dem is not None,
                "mesh_height_surface": mesh_meta.get("default_surface", "predicted_agl"),
                "available_mesh_surfaces": available_surfaces if generate_3d else [],
                "mesh_horizontal_units": agl_scene_obj.horizontal_units,
                "vertical_exaggeration_baked_into_artifacts": 1.0,
            },
            "height_stats": {
                "type": "AGL",
                "unit": "metres",
                "min_m": float(np.min(predicted_agl)),
                "max_m": float(np.max(predicted_agl)),
                "mean_m": float(np.mean(predicted_agl)),
                "median_m": float(np.median(predicted_agl)),
                "std_m": float(np.std(predicted_agl)),
            },
            "slope_mean_deg": float(np.mean(slope_deg)) if slope_deg is not None else None,
            "slope_max_deg": float(np.max(slope_deg)) if slope_deg is not None else None,
            "products": products,
            "artifacts": {
                "predicted_agl_npy": f"/api/v1/scenes/{scene_id}/download/predicted_agl.npy",
                "predicted_agl_tif": (
                    f"/api/v1/scenes/{scene_id}/download/predicted_agl.tif"
                    if products["predicted_agl_tif"]
                    else None
                ),
                "relative_surface_npy": (
                    f"/api/v1/scenes/{scene_id}/download/relative_surface.npy"
                    if products["relative_surface_npy"]
                    else None
                ),
                "relative_surface_png": (
                    f"/api/v1/scenes/{scene_id}/relative_surface"
                    if products["relative_surface_png"]
                    else None
                ),
                "aligned_terrain_dem_tif": (
                    f"/api/v1/scenes/{scene_id}/download/aligned_terrain_dem.tif"
                    if products["aligned_terrain_dem_tif"]
                    else None
                ),
                "absolute_dsm_tif": (
                    f"/api/v1/scenes/{scene_id}/download/absolute_dsm.tif"
                    if products["absolute_dsm_tif"]
                    else None
                ),
                "mesh_glb": f"/api/v1/scenes/{scene_id}/mesh.glb" if generate_3d else None,
                "mesh_agl_glb": (
                    f"/api/v1/scenes/{scene_id}/mesh.glb?surface=agl"
                    if generate_3d
                    else None
                ),
                "mesh_relative_glb": (
                    f"/api/v1/scenes/{scene_id}/mesh.glb?surface=relative"
                    if products["mesh_relative_glb"]
                    else None
                ),
                "mesh_absolute_dsm_glb": (
                    f"/api/v1/scenes/{scene_id}/mesh.glb?surface=absolute_dsm"
                    if generate_3d and base_dem is not None
                    else None
                ),
                "pointcloud_ply": (
                    f"/api/v1/scenes/{scene_id}/pointcloud.ply" if generate_pointcloud else None
                ),
                "metadata": f"/api/v1/scenes/{scene_id}/metadata",
                "texture_rgb": f"/api/v1/scenes/{scene_id}/texture/rgb",
                "heightmap": f"/api/v1/scenes/{scene_id}/heightmap",
                "relative_surface": (
                    f"/api/v1/scenes/{scene_id}/relative_surface"
                    if products["texture_relative_surface"]
                    else None
                ),
                "slope": f"/api/v1/scenes/{scene_id}/slope" if slope_deg is not None else None,
            },
            "processing_timings": timings,
            "peak_vram_mb": round(float(peak_vram_mb), 3),
            "warnings": warnings,
            "mesh_info": mesh_meta,
            "relative_surface_mesh_info": relative_mesh_meta,
            "absolute_surface_mesh_info": absolute_mesh_meta,
            "pointcloud_info": ply_meta,
        }
        metadata["processing_timings"]["scene_products_s"] = round(
            time.perf_counter() - t0, 6
        )
        metadata["artifact_sizes_bytes"] = self._artifact_sizes(scene_dir)
        self._write_metadata(scene_id, metadata)
        return metadata

    def finalize_scene_runtime(
        self, scene_id: str, full_api_latency_s: float, total_pipeline_s: float
    ) -> dict:
        metadata = self.load_scene_metadata(scene_id)
        if metadata is None:
            raise FileNotFoundError(f"Scene metadata missing for {scene_id}")
        metadata.setdefault("processing_timings", {})["full_api_latency_s"] = round(
            float(full_api_latency_s), 6
        )
        metadata["processing_timings"]["total_pipeline_s"] = round(
            float(total_pipeline_s), 6
        )
        metadata["artifact_sizes_bytes"] = self._artifact_sizes(self.get_scene_dir(scene_id))
        self._write_metadata(scene_id, metadata)
        return metadata

    def inspect_pixel(self, scene_id: str, u: int, v: int) -> Optional[dict]:
        scene_dir = self.get_scene_dir(scene_id)
        agl_path = os.path.join(scene_dir, "rasters", "predicted_agl.npy")
        if not os.path.exists(agl_path):
            return None
        agl_arr = np.load(agl_path)
        height, width = agl_arr.shape
        u = int(np.clip(u, 0, width - 1))
        v = int(np.clip(v, 0, height - 1))

        dsm_path = os.path.join(scene_dir, "rasters", "absolute_dsm.npy")
        slope_path = os.path.join(scene_dir, "rasters", "slope.npy")
        metadata = self.load_scene_metadata(scene_id) or {}
        spatial = metadata.get("spatial_info", {})
        gsd = spatial.get("gsd_m")
        world_coordinates = None
        if gsd is not None:
            world_coordinates = {
                "x": (u - width / 2.0) * float(gsd),
                "z": (height / 2.0 - v) * float(gsd),
            }

        return {
            "scene_id": scene_id,
            "pixel": [u, v],
            "predicted_agl_m": float(agl_arr[v, u]),
            "absolute_elevation_m": (
                float(np.load(dsm_path, mmap_mode="r")[v, u])
                if os.path.exists(dsm_path)
                else None
            ),
            "semantic_class": None,
            "slope_deg": (
                float(np.load(slope_path, mmap_mode="r")[v, u])
                if os.path.exists(slope_path)
                else None
            ),
            "world_coordinates_m": world_coordinates,
            "is_georeferenced": bool(spatial.get("is_georeferenced", False)),
            "warnings": metadata.get("warnings", []),
        }

    def measure_distance(
        self, scene_id: str, pt_a: list, pt_b: list, is_pixel: bool = True
    ) -> Optional[dict]:
        scene_dir = self.get_scene_dir(scene_id)
        agl_path = os.path.join(scene_dir, "rasters", "predicted_agl.npy")
        if not os.path.exists(agl_path):
            return None
        metadata = self.load_scene_metadata(scene_id) or {}
        gsd = metadata.get("spatial_info", {}).get("gsd_m")
        if gsd is None:
            raise ValueError(GSD_WARNING)
        gsd = float(gsd)

        agl_arr = np.load(agl_path)
        height, width = agl_arr.shape
        if is_pixel:
            u1 = int(np.clip(pt_a[0], 0, width - 1))
            v1 = int(np.clip(pt_a[1], 0, height - 1))
            u2 = int(np.clip(pt_b[0], 0, width - 1))
            v2 = int(np.clip(pt_b[1], 0, height - 1))
            x1, z1, y1 = (u1 - width / 2.0) * gsd, (height / 2.0 - v1) * gsd, float(agl_arr[v1, u1])
            x2, z2, y2 = (u2 - width / 2.0) * gsd, (height / 2.0 - v2) * gsd, float(agl_arr[v2, u2])
        else:
            x1, y1, z1 = pt_a
            x2, y2, z2 = pt_b

        horizontal_distance = float(np.hypot(x2 - x1, z2 - z1))
        elevation_delta = float(y2 - y1)
        direct_distance = float(np.hypot(horizontal_distance, elevation_delta))
        slope_angle = float(
            np.rad2deg(np.arctan2(abs(elevation_delta), max(1e-4, horizontal_distance)))
        )
        return {
            "scene_id": scene_id,
            "horizontal_distance_m": horizontal_distance,
            "direct_3d_distance_m": direct_distance,
            "elevation_delta_m": elevation_delta,
            "slope_angle_deg": slope_angle,
            "start_elevation_m": float(y1),
            "end_elevation_m": float(y2),
            "warnings": metadata.get("warnings", []),
        }


scene_service = SceneService()
