"""Real M2-FINAL API/product integration matrix (no training or model mutation)."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import Affine, from_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.main import app
from backend.app.config import settings


EXPECTED_SHA = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"
GSD_WARNING = "GSD unavailable: horizontal metric distance and slope disabled."
DSM_WARNING = "Absolute DSM unavailable: aligned terrain DEM was not supplied."


def source_rgb() -> np.ndarray:
    source = ROOT / "data/gamus/dataset/images/test/NYC_00735_IMG.h5"
    with h5py.File(source, "r") as handle:
        array = handle[next(iter(handle.keys()))][:]
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.max() <= 1.5:
        array = np.clip(array * 255.0, 0, 255)
    return array.astype(np.uint8)


def write_rgb_geotiff(path: Path, rgb: np.ndarray, transform: Affine):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rgb.shape[0],
        width=rgb.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:32618",
        transform=transform,
    ) as dataset:
        dataset.write(np.transpose(rgb, (2, 0, 1)))


def write_dem(path: Path, shape: tuple[int, int], transform: Affine, value: float):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32618",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.full(shape, value, dtype=np.float32), 1)


def poll(client: TestClient, job_id: str, timeout_s: float = 180.0) -> dict:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        response = client.get(f"/api/v1/jobs/{job_id}")
        response.raise_for_status()
        record = response.json()
        if record["status"] in {"completed", "failed"}:
            return record
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} timed out after {timeout_s}s")


def submit(
    client: TestClient,
    image_path: Path,
    scene_id: str,
    dem_path: Path | None = None,
    gsd_m: float | None = None,
) -> dict:
    files = {
        "file": (
            image_path.name,
            image_path.read_bytes(),
            "image/tiff" if image_path.suffix.lower() in {".tif", ".tiff"} else "image/png",
        )
    }
    if dem_path is not None:
        files["dem_file"] = (dem_path.name, dem_path.read_bytes(), "image/tiff")
    data = {
        "scene_name": scene_id,
        "generate_3d": "true",
        "generate_pointcloud": "false",
        "mesh_resolution": "256",
        "vertical_exaggeration": "1.0",
    }
    if gsd_m is not None:
        data["gsd_m"] = str(gsd_m)
    response = client.post("/api/v1/jobs", files=files, data=data)
    if response.status_code != 202:
        return {
            "submission_status_code": response.status_code,
            "submission_body": response.json(),
            "status": "submission_failed",
        }
    return poll(client, response.json()["job_id"])


def verify_success(
    client: TestClient,
    final_job: dict,
    expected_shape: tuple[int, int],
) -> dict:
    assert final_job["status"] == "completed", final_job
    scene_id = final_job["scene_id"]
    metadata_response = client.get(f"/api/v1/scenes/{scene_id}")
    metadata_response.raise_for_status()
    metadata = metadata_response.json()

    scene_dir = Path(settings.STORAGE_DIR) / scene_id
    agl = np.load(scene_dir / "rasters/predicted_agl.npy")
    assert agl.shape == expected_shape
    assert agl.dtype == np.float32
    assert np.isfinite(agl).all() and float(agl.min()) >= 0.0
    assert metadata["model"]["identity"] == "M2-FINAL"
    assert metadata["model"]["checkpoint_sha256"] == EXPECTED_SHA
    assert metadata["output_semantics"]["primary"] == "predicted_agl_ndsm"
    assert metadata["output_semantics"]["vertical_exaggeration_baked_into_artifacts"] == 1.0
    assert metadata["products"]["mesh_glb"] is True

    glb_response = client.get(f"/api/v1/scenes/{scene_id}/mesh.glb")
    glb_response.raise_for_status()
    assert glb_response.content[:4] == b"glTF"
    loaded = trimesh.load(io.BytesIO(glb_response.content), file_type="glb", force="scene")
    assert isinstance(loaded, trimesh.Scene) and len(loaded.geometry) >= 1
    texture_kinds = sorted(
        {getattr(geometry.visual, "kind", None) for geometry in loaded.geometry.values()}
    )
    assert "texture" in texture_kinds

    inspect_response = client.post(
        f"/api/v1/scenes/{scene_id}/inspect",
        json={"u": expected_shape[1] // 2, "v": expected_shape[0] // 2},
    )
    inspect_response.raise_for_status()
    inspection = inspect_response.json()
    assert inspection["predicted_agl_m"] >= 0.0
    assert inspection["semantic_class"] is None

    required_timings = {
        "dav2_inference_s",
        "m2_inference_s",
        "total_ai_s",
        "mesh_generation_s",
        "glb_generation_s",
        "full_api_latency_s",
    }
    assert required_timings.issubset(metadata["processing_timings"])
    assert metadata["artifact_sizes_bytes"]["mesh_glb"] == len(glb_response.content)

    return {
        "status": "PASS",
        "scene_id": scene_id,
        "shape": list(agl.shape),
        "height_min_m": float(agl.min()),
        "height_max_m": float(agl.max()),
        "height_mean_m": float(agl.mean()),
        "warnings": metadata["warnings"],
        "spatial_info": metadata["spatial_info"],
        "output_semantics": metadata["output_semantics"],
        "processing_timings": metadata["processing_timings"],
        "peak_vram_mb": metadata["peak_vram_mb"],
        "artifact_sizes_bytes": metadata["artifact_sizes_bytes"],
        "glb_geometry_count": len(loaded.geometry),
        "glb_texture_kinds": texture_kinds,
    }


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = ROOT / "outputs/m2_product_integration" / timestamp
    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=False)

    rgb = source_rgb()
    png_rgb = rgb[:1024, :1024]
    jpg_rgb = rgb[200:533, 100:801]
    geotiff_rgb = rgb[400:912, 400:912]
    metadata_absent_rgb = rgb[:256, :320]

    png_path = fixture_dir / "normal_rgb.png"
    jpg_path = fixture_dir / "arbitrary_rgb.jpg"
    geotiff_path = fixture_dir / "valid_rgb.tif"
    aligned_dem_path = fixture_dir / "aligned_dem.tif"
    shifted_dem_path = fixture_dir / "shifted_dem.tif"
    no_metadata_path = fixture_dir / "metadata_absent.tif"
    corrupt_path = fixture_dir / "corrupt.png"

    Image.fromarray(png_rgb).save(png_path)
    Image.fromarray(jpg_rgb).save(jpg_path, quality=92)
    transform = from_origin(584000.0, 4510000.0, 0.5, 0.5)
    write_rgb_geotiff(geotiff_path, geotiff_rgb, transform)
    write_dem(aligned_dem_path, geotiff_rgb.shape[:2], transform, 12.5)
    write_dem(
        shifted_dem_path,
        geotiff_rgb.shape[:2],
        from_origin(584001.0, 4510000.0, 0.5, 0.5),
        12.5,
    )
    Image.fromarray(metadata_absent_rgb).save(no_metadata_path)
    corrupt_path.write_bytes(b"not an image")

    suffix = timestamp[-6:]
    results = {
        "test_name": "M2-FINAL real product integration matrix",
        "timestamp": timestamp,
        "checkpoint_sha256_expected": EXPECTED_SHA,
        "cases": {},
    }

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model"]["name"].startswith("M2-FINAL")
        assert health["model"]["checkpoint_sha256_verified"] is True
        assert health["model"]["checkpoint_sha256"] == EXPECTED_SHA
        assert health["model"]["dav2_frozen"] is True
        results["health"] = health

        success_specs = [
            ("normal_png_unknown_gsd", png_path, None, None, png_rgb.shape[:2]),
            ("arbitrary_jpg_user_gsd", jpg_path, None, 0.5, jpg_rgb.shape[:2]),
            ("valid_geotiff", geotiff_path, None, None, geotiff_rgb.shape[:2]),
            ("geotiff_aligned_dem", geotiff_path, aligned_dem_path, None, geotiff_rgb.shape[:2]),
            ("metadata_absent_tiff", no_metadata_path, None, None, metadata_absent_rgb.shape[:2]),
        ]
        for case_name, image_path, dem_path, gsd_m, expected_shape in success_specs:
            scene_id = f"m2int_{case_name}_{suffix}"
            final_job = submit(client, image_path, scene_id, dem_path, gsd_m)
            verified = verify_success(client, final_job, expected_shape)
            warnings = verified["warnings"]
            if case_name in {"normal_png_unknown_gsd", "metadata_absent_tiff"}:
                assert GSD_WARNING in warnings
                assert verified["spatial_info"]["gsd_m"] is None
            if case_name == "normal_png_unknown_gsd":
                assert verified["processing_timings"]["window_count"] == 9
                assert verified["processing_timings"]["window_size_px"] == 512
                assert verified["processing_timings"]["window_step_px"] == 256
            if case_name == "arbitrary_jpg_user_gsd":
                assert GSD_WARNING not in warnings
                assert verified["spatial_info"]["gsd_source"] == "user"
            if case_name == "valid_geotiff":
                assert verified["spatial_info"]["crs"] == "EPSG:32618"
                assert verified["spatial_info"]["transform"][:6] == list(transform)[:6]
            if case_name == "geotiff_aligned_dem":
                assert DSM_WARNING not in warnings
                scene_dir = Path(settings.STORAGE_DIR) / verified["scene_id"]
                agl = np.load(scene_dir / "rasters/predicted_agl.npy")
                dsm = np.load(scene_dir / "rasters/absolute_dsm.npy")
                assert np.allclose(dsm - agl, 12.5, atol=1e-5)
                assert verified["output_semantics"]["mesh_height_surface"] == "predicted_agl"
                assert verified["output_semantics"]["available_mesh_surfaces"] == [
                    "predicted_agl",
                    "absolute_dsm",
                ]
                agl_mesh = client.get(
                    f"/api/v1/scenes/{verified['scene_id']}/mesh.glb?surface=agl"
                )
                absolute_mesh = client.get(
                    f"/api/v1/scenes/{verified['scene_id']}/mesh.glb?surface=absolute_dsm"
                )
                assert agl_mesh.status_code == 200
                assert absolute_mesh.status_code == 200
                assert agl_mesh.content != absolute_mesh.content
            else:
                assert DSM_WARNING in warnings
            results["cases"][case_name] = verified

        corrupt_job = submit(client, corrupt_path, f"m2int_corrupt_{suffix}")
        assert corrupt_job["status"] == "failed"
        assert "corrupt or unreadable" in corrupt_job["error_message"].lower()
        results["cases"]["invalid_file"] = {
            "status": "PASS",
            "observed_job_status": corrupt_job["status"],
            "error_message": corrupt_job["error_message"],
            "full_api_latency_s": corrupt_job["execution_time_s"],
        }

        mismatch_job = submit(
            client,
            geotiff_path,
            f"m2int_misaligned_dem_{suffix}",
            shifted_dem_path,
        )
        assert mismatch_job["status"] == "failed"
        assert "DEM alignment validation failed" in mismatch_job["error_message"]
        results["cases"]["misaligned_dem"] = {
            "status": "PASS",
            "observed_job_status": mismatch_job["status"],
            "error_message": mismatch_job["error_message"],
            "full_api_latency_s": mismatch_job["execution_time_s"],
        }

    results["cuda_available"] = torch.cuda.is_available()
    results["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    results["overall_status"] = "PASS"
    result_path = output_dir / "integration_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"overall_status": "PASS", "result_path": str(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
