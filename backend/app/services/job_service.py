"""Asynchronous product pipeline for RGB/GeoTIFF to M2 AGL and 3D artifacts."""

import os
import threading
import time
import uuid
from typing import Dict, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.app.schemas.jobs import JobStatus
from backend.app.services.inference_service import inference_service
from backend.app.services.geospatial_service import geospatial_service
from backend.app.services.scene_service import scene_service


class JobService:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "scene_id": None,
                "status": JobStatus.QUEUED,
                "progress_pct": 5,
                "current_stage": "Job queued in background manager",
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "error_message": None,
                "execution_time_s": None,
                "model_identity": "M2-FINAL",
                "warnings": [],
                "processing_timings": None,
                "peak_vram_mb": None,
                "artifact_sizes_bytes": None,
                "_created_perf": time.perf_counter(),
            }
        return job_id

    def get_job_status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)

    def run_pipeline_async(
        self,
        job_id: str,
        file_path: str,
        is_geotiff: bool,
        dem_file_path: Optional[str] = None,
        user_gsd_m: Optional[float] = None,
        generate_3d: bool = True,
        generate_pointcloud: bool = True,
        mesh_resolution: int = 384,
        vertical_exaggeration: float = 1.0,
        custom_scene_id: Optional[str] = None,
    ):
        thread = threading.Thread(
            target=self._execute_pipeline,
            args=(
                job_id,
                file_path,
                is_geotiff,
                dem_file_path,
                user_gsd_m,
                generate_3d,
                generate_pointcloud,
                mesh_resolution,
                vertical_exaggeration,
                custom_scene_id,
            ),
            daemon=True,
        )
        thread.start()

    def _execute_pipeline(
        self,
        job_id: str,
        file_path: str,
        is_geotiff: bool,
        dem_file_path: Optional[str],
        user_gsd_m: Optional[float],
        generate_3d: bool,
        generate_pointcloud: bool,
        mesh_resolution: int,
        vertical_exaggeration: float,
        custom_scene_id: Optional[str],
    ):
        pipeline_start = time.perf_counter()
        scene_id = custom_scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        job_record = self.get_job_status(job_id) or {}
        created_perf = float(job_record.get("_created_perf", pipeline_start))
        self.update_job(
            job_id,
            scene_id=scene_id,
            status=JobStatus.VALIDATING,
            progress_pct=15,
            current_stage="Validating input image and spatial metadata",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        try:
            if scene_service.scene_exists(scene_id):
                raise ValueError(
                    f"Scene '{scene_id}' already exists; choose a unique scene name"
                )

            ingestion_start = time.perf_counter()
            if is_geotiff:
                spatial_meta = geospatial_service.parse_geotiff(file_path)
                rgb_arr = spatial_meta.pop("rgb_uint8")
                spatial_meta["format"] = "GeoTIFF"
                if not spatial_meta.get("is_georeferenced") and user_gsd_m is not None:
                    spatial_meta["gsd_m"] = float(user_gsd_m)
                    spatial_meta["gsd_source"] = "user"
                    spatial_meta["horizontal_units"] = "metres"
            else:
                try:
                    with Image.open(file_path) as image:
                        image.verify()
                    with Image.open(file_path) as image:
                        rgb_arr = np.array(image.convert("RGB"))
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise ValueError(f"Input image is corrupt or unreadable: {exc}") from exc
                height, width, _ = rgb_arr.shape
                spatial_meta = {
                    "format": "RGB Image",
                    "is_georeferenced": False,
                    "crs": None,
                    "transform": None,
                    "bounds": None,
                    "width": width,
                    "height": height,
                    "gsd_m": float(user_gsd_m) if user_gsd_m is not None else None,
                    "gsd_source": "user" if user_gsd_m is not None else "unavailable",
                    "horizontal_units": "metres" if user_gsd_m is not None else None,
                    "resolution": None,
                }
            ingestion_time = time.perf_counter() - ingestion_start

            self.update_job(
                job_id,
                status=JobStatus.RUNNING_INFERENCE,
                progress_pct=40,
                current_stage="Running frozen DAV2 Small + M2-FINAL AGL inference",
            )
            infer_result = inference_service.predict_height_map(rgb_arr)
            predicted_agl = infer_result["predicted_agl_m"]

            self.update_job(
                job_id,
                status=JobStatus.PROCESSING_GEOSPATIAL,
                progress_pct=65,
                current_stage="Validating geospatial scale and optional DEM alignment",
            )
            geospatial_start = time.perf_counter()
            base_dem = None
            if dem_file_path:
                dsm_result = geospatial_service.align_and_compute_dsm(
                    predicted_agl, dem_file_path, spatial_meta
                )
                base_dem = dsm_result["base_dem"]

            scene_dir = scene_service.get_scene_dir(scene_id)
            raster_dir = os.path.join(scene_dir, "rasters")
            os.makedirs(raster_dir, exist_ok=True)
            if spatial_meta.get("is_georeferenced"):
                geospatial_service.export_geotiff(
                    predicted_agl,
                    os.path.join(raster_dir, "predicted_agl.tif"),
                    spatial_meta,
                )
                if base_dem is not None:
                    geospatial_service.export_geotiff(
                        base_dem + predicted_agl,
                        os.path.join(raster_dir, "absolute_dsm.tif"),
                        spatial_meta,
                    )
            geospatial_time = time.perf_counter() - geospatial_start

            self.update_job(
                job_id,
                status=JobStatus.GENERATING_3D,
                progress_pct=80,
                current_stage="Generating physical 1.0x textured AGL/DSM surface mesh and GLB",
            )
            timings = {
                "input_validation_s": round(ingestion_time, 6),
                "dav2_inference_s": round(float(infer_result["dav2_time_s"]), 6),
                "m2_inference_s": round(float(infer_result["m2_time_s"]), 6),
                "total_ai_s": round(float(infer_result["ai_time_s"]), 6),
                "inference_service_total_s": round(
                    float(infer_result["total_inference_time_s"]), 6
                ),
                "geospatial_processing_s": round(geospatial_time, 6),
                "window_count": int(infer_result["window_count"]),
                "window_size_px": int(infer_result["window_size_px"]),
                "window_step_px": int(infer_result["window_step_px"]),
            }
            metadata = scene_service.create_scene_products(
                scene_id=scene_id,
                rgb_image=rgb_arr,
                predicted_agl=predicted_agl,
                spatial_meta=spatial_meta,
                base_dem=base_dem,
                generate_3d=generate_3d,
                generate_pointcloud=generate_pointcloud,
                mesh_resolution=mesh_resolution,
                vertical_exaggeration=vertical_exaggeration,
                processing_timings=timings,
                peak_vram_mb=float(infer_result["peak_vram_mb"]),
            )

            total_pipeline_s = time.perf_counter() - pipeline_start
            full_api_latency_s = time.perf_counter() - created_perf
            metadata = scene_service.finalize_scene_runtime(
                scene_id, full_api_latency_s, total_pipeline_s
            )
            self.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress_pct=100,
                current_stage="M2-FINAL AGL scene and 3D products completed",
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                execution_time_s=round(full_api_latency_s, 3),
                warnings=metadata.get("warnings", []),
                processing_timings=metadata.get("processing_timings"),
                peak_vram_mb=metadata.get("peak_vram_mb"),
                artifact_sizes_bytes=metadata.get("artifact_sizes_bytes"),
            )
        except Exception as exc:
            full_api_latency_s = time.perf_counter() - created_perf
            print(f"[JobService] Job {job_id} failed: {exc}")
            self.update_job(
                job_id,
                status=JobStatus.FAILED,
                progress_pct=100,
                current_stage="Processing failed",
                error_message=str(exc),
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                execution_time_s=round(full_api_latency_s, 3),
            )


job_service = JobService()
