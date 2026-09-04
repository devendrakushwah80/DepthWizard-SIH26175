"""
DepthWizard (SIH26175) — Phase 1 Requirement Closure Test Matrix
Covers Cases A through H + M2-FINAL Checkpoint SHA-256 Integrity Verification.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from backend.app.config import settings
from backend.app.schemas.jobs import JobStatus
from backend.app.services.job_service import job_service
from backend.app.services.scene_service import scene_service


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers to generate synthetic test inputs
# ---------------------------------------------------------------------------

def _create_png(path: Path, width: int = 64, height: int = 64) -> Path:
    yy, xx = np.mgrid[:height, :width]
    arr = np.stack([xx * 3, yy * 3, np.full_like(xx, 180)], axis=-1).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def _create_jpg(path: Path, width: int = 64, height: int = 64) -> Path:
    yy, xx = np.mgrid[:height, :width]
    arr = np.stack([xx * 3, yy * 3, np.full_like(xx, 180)], axis=-1).astype(np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=90)
    return path


def _create_geotiff(
    path: Path,
    width: int = 64,
    height: int = 64,
    crs: str = "EPSG:32618",
    origin_x: float = 500000.0,
    origin_y: float = 4500000.0,
    res: float = 0.5,
) -> Path:
    transform = from_origin(origin_x, origin_y, res, res)
    yy, xx = np.mgrid[:height, :width]
    data = np.stack([xx * 3, yy * 3, np.full_like(xx, 150)], axis=0).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def _create_dem_geotiff(
    path: Path,
    width: int = 64,
    height: int = 64,
    crs: str = "EPSG:32618",
    origin_x: float = 500000.0,
    origin_y: float = 4500000.0,
    res: float = 0.5,
    base_elev: float = 25.0,
) -> Path:
    transform = from_origin(origin_x, origin_y, res, res)
    yy, xx = np.mgrid[:height, :width]
    data = (base_elev + 0.1 * xx + 0.05 * yy).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


# ---------------------------------------------------------------------------
# Test Cases A through H + SHA256 Integrity
# ---------------------------------------------------------------------------

def test_case_a_non_geo_png_without_user_gsd(tmp_path):
    """Case A: Non-georeferenced PNG without user GSD.
    Must produce relative surface [0, 1], predicted AGL in metres, and pixel units (no metric GSD).
    """
    img_path = _create_png(tmp_path / "case_a.png")
    job_id = job_service.create_job()
    scene_id = f"test_a_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(img_path),
            is_geotiff=False,
            dem_file_path=None,
            user_gsd_m=None,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED

        meta = scene_service.load_scene_metadata(scene_id)
        assert meta is not None

        # Spatial assertions: No GSD, non-georeferenced
        spatial = meta["spatial_info"]
        assert spatial["is_georeferenced"] is False
        assert spatial["gsd_m"] is None
        assert spatial["gsd_source"] in (None, "unavailable")
        assert spatial["crs"] is None

        # Products assertions
        prods = meta["products"]
        assert prods["relative_surface_npy"] is True
        assert prods["mesh_relative_glb"] is True
        assert prods["predicted_agl_npy"] is True
        assert prods["mesh_agl_glb"] is True
        assert prods["relative_surface_png"] is True
        assert prods["predicted_agl_tif"] is False
        assert prods["absolute_dsm_tif"] is False

        # Relative surface raster range [0, 1]
        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        rel_path = scene_dir / "rasters" / "relative_surface.npy"
        assert rel_path.exists()
        rel = np.load(rel_path)
        assert rel.min() >= -1e-6
        assert rel.max() <= 1.0 + 1e-6

        # Pixel inspection: world coordinates must be None
        insp = scene_service.inspect_pixel(scene_id, 16, 16)
        assert insp["world_coordinates_m"] is None
        assert insp["predicted_agl_m"] is not None
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_b_non_geo_png_with_user_gsd(tmp_path):
    """Case B: Non-georeferenced PNG with user-specified GSD.
    Must record user GSD and provide metric ground scale coordinates.
    """
    img_path = _create_png(tmp_path / "case_b.png", width=100, height=80)
    job_id = job_service.create_job()
    scene_id = f"test_b_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(img_path),
            is_geotiff=False,
            dem_file_path=None,
            user_gsd_m=0.25,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED

        meta = scene_service.load_scene_metadata(scene_id)
        assert meta is not None
        spatial = meta["spatial_info"]
        assert spatial["is_georeferenced"] is False
        assert spatial["gsd_m"] == pytest.approx(0.25)
        assert spatial["gsd_source"] == "user"
        assert spatial["physical_width_m"] == pytest.approx(100 * 0.25)
        assert spatial["physical_height_m"] == pytest.approx(80 * 0.25)

        insp = scene_service.inspect_pixel(scene_id, 20, 10)
        assert insp["world_coordinates_m"] is not None
        expected_x = (20 - 100 / 2.0) * 0.25
        expected_z = (80 / 2.0 - 10) * 0.25
        assert insp["world_coordinates_m"]["x"] == pytest.approx(expected_x)
        assert insp["world_coordinates_m"]["z"] == pytest.approx(expected_z)
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_c_non_geo_jpg(tmp_path):
    """Case C: Non-georeferenced JPG."""
    img_path = _create_jpg(tmp_path / "case_c.jpg")
    job_id = job_service.create_job()
    scene_id = f"test_c_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(img_path),
            is_geotiff=False,
            dem_file_path=None,
            user_gsd_m=None,
            generate_3d=False,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED
        meta = scene_service.load_scene_metadata(scene_id)
        assert meta is not None
        assert meta["input_info"]["format"].upper() in ["JPEG", "JPG", "RGB IMAGE"]
        assert meta["products"]["relative_surface_npy"] is True
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_d_geotiff_alone(tmp_path):
    """Case D: Georeferenced GeoTIFF alone (no terrain DEM).
    Must export predicted_agl.tif preserving CRS and affine transform, and not generate absolute DSM.
    """
    tif_path = _create_geotiff(tmp_path / "case_d.tif", width=64, height=64, crs="EPSG:32618", res=0.5)
    job_id = job_service.create_job()
    scene_id = f"test_d_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(tif_path),
            is_geotiff=True,
            dem_file_path=None,
            user_gsd_m=None,
            generate_3d=False,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED
        meta = scene_service.load_scene_metadata(scene_id)
        assert meta is not None
        spatial = meta["spatial_info"]
        assert spatial["is_georeferenced"] is True
        assert spatial["crs"] == "EPSG:32618"
        prods = meta["products"]
        assert prods["predicted_agl_tif"] is True
        assert prods["absolute_dsm_tif"] is False
        assert prods["aligned_terrain_dem_tif"] is False

        # Check exported GeoTIFF
        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        agl_tif = scene_dir / "rasters" / "predicted_agl.tif"
        assert agl_tif.exists()
        with rasterio.open(agl_tif) as src:
            assert src.crs.to_string() == "EPSG:32618"
            assert src.res == (0.5, 0.5)
            assert src.shape == (64, 64)
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_e_geotiff_with_matching_dem(tmp_path):
    """Case E: Georeferenced GeoTIFF + matching 0.5m DEM.
    Must export predicted_agl.tif, aligned_terrain_dem.tif, and absolute_dsm.tif.
    Must enforce numeric identity: absolute_dsm - aligned_dem == predicted_agl.
    """
    tif_path = _create_geotiff(tmp_path / "case_e_opt.tif", width=64, height=64, res=0.5)
    dem_path = _create_dem_geotiff(tmp_path / "case_e_dem.tif", width=64, height=64, res=0.5, base_elev=40.0)

    job_id = job_service.create_job()
    scene_id = f"test_e_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(tif_path),
            is_geotiff=True,
            dem_file_path=str(dem_path),
            user_gsd_m=None,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED
        meta = scene_service.load_scene_metadata(scene_id)
        assert meta is not None
        prods = meta["products"]
        assert prods["predicted_agl_tif"] is True
        assert prods["aligned_terrain_dem_tif"] is True
        assert prods["absolute_dsm_tif"] is True
        assert prods["mesh_absolute_dsm_glb"] is True

        # Numeric identity assertion on exported files
        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        with rasterio.open(scene_dir / "rasters" / "predicted_agl.tif") as s_agl:
            agl = s_agl.read(1)
        with rasterio.open(scene_dir / "rasters" / "aligned_terrain_dem.tif") as s_dem:
            dem = s_dem.read(1)
        with rasterio.open(scene_dir / "rasters" / "absolute_dsm.tif") as s_dsm:
            dsm = s_dsm.read(1)

        np.testing.assert_allclose(dsm - dem, agl, atol=1e-4)

        # DEM provenance verification
        prov = meta["dem_provenance"]
        assert prov is not None
        assert prov["coverage_percentage"] >= 99.9
        assert prov["resampling_method"] in ("identity", "bilinear")
        assert "Unverified" in prov["vertical_datum"] or "not verified" in prov["vertical_datum"].lower()
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_f_geotiff_with_coarse_30m_srtm_dem(tmp_path):
    """Case F: Georeferenced GeoTIFF + coarse 30m SRTM DEM.
    Must resample coarse DEM onto optical grid (0.5m), enforce identity, and track provenance.
    """
    # Optical: 64x64 at 0.5m = 32m x 32m extent
    tif_path = _create_geotiff(
        tmp_path / "case_f_opt.tif",
        width=64,
        height=64,
        origin_x=500000.0,
        origin_y=4500000.0,
        res=0.5,
    )
    # Coarse DEM: 8x8 at 30m = 240m x 240m extent covering the optical region with extra padding
    dem_path = _create_dem_geotiff(
        tmp_path / "case_f_dem_coarse.tif",
        width=8,
        height=8,
        origin_x=499900.0,
        origin_y=4500100.0,
        res=30.0,
        base_elev=150.0,
    )

    job_id = job_service.create_job()
    scene_id = f"test_f_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(tif_path),
            is_geotiff=True,
            dem_file_path=str(dem_path),
            user_gsd_m=None,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED
        meta = scene_service.load_scene_metadata(scene_id)
        assert meta is not None

        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        with rasterio.open(scene_dir / "rasters" / "predicted_agl.tif") as s_agl:
            agl = s_agl.read(1)
            assert s_agl.res == (0.5, 0.5)
        with rasterio.open(scene_dir / "rasters" / "aligned_terrain_dem.tif") as s_dem:
            dem = s_dem.read(1)
            assert s_dem.res == (0.5, 0.5)
            assert dem.shape == (64, 64)
        with rasterio.open(scene_dir / "rasters" / "absolute_dsm.tif") as s_dsm:
            dsm = s_dsm.read(1)

        # Mathematical identity must hold even when base DEM is resampled from coarse 30m
        np.testing.assert_allclose(dsm - dem, agl, atol=1e-4)
        prov = meta["dem_provenance"]
        assert prov["coverage_percentage"] >= 99.0
        assert prov["original_dem_resolution"] == [30.0, 30.0]
        assert prov["resampling_method"] == "bilinear"
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_g_non_overlapping_dem_rejection(tmp_path):
    """Case G: Non-overlapping DEM rejection."""
    tif_path = _create_geotiff(
        tmp_path / "case_g_opt.tif",
        origin_x=500000.0,
        origin_y=4500000.0,
        res=0.5,
    )
    # Disjoint DEM far away
    dem_path = _create_dem_geotiff(
        tmp_path / "case_g_dem_disjoint.tif",
        origin_x=600000.0,
        origin_y=4600000.0,
        res=0.5,
    )

    job_id = job_service.create_job()
    scene_id = f"test_g_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(tif_path),
            is_geotiff=True,
            dem_file_path=str(dem_path),
            user_gsd_m=None,
            generate_3d=False,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.FAILED
        assert "overlap" in job["error_message"].lower() or "disjoint" in job["error_message"].lower()
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_h_corrupted_image_rejection(tmp_path):
    """Case H: Corrupted image rejection."""
    corrupt_file = tmp_path / "corrupted.png"
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupted_payload")

    job_id = job_service.create_job()
    scene_id = f"test_h_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(corrupt_file),
            is_geotiff=False,
            dem_file_path=None,
            user_gsd_m=None,
            generate_3d=False,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.FAILED
        assert job["error_message"] is not None
    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_m2_final_checkpoint_integrity():
    """Verify that M2-FINAL checkpoint SHA-256 matches the officially sealed constant."""
    expected_sha = settings.M2_MODEL_CHECKPOINT_SHA256
    ckpt_path = Path(settings.M2_MODEL_CHECKPOINT)

    assert ckpt_path.exists(), f"M2-FINAL checkpoint not found at {ckpt_path}"

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()

    assert actual_sha == expected_sha, (
        f"M2-FINAL checkpoint SHA-256 mismatch!\nExpected: {expected_sha}\nActual:   {actual_sha}"
    )


def test_m3_final_checkpoint_integrity():
    """Verify that promoted M3-FINAL checkpoint SHA-256 matches the official constant."""
    expected_sha = settings.M3_MODEL_CHECKPOINT_SHA256
    ckpt_path = Path(settings.M3_MODEL_CHECKPOINT)

    assert ckpt_path.exists(), f"M3-FINAL checkpoint not found at {ckpt_path}"

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()

    assert actual_sha == expected_sha, (
        f"M3-FINAL checkpoint SHA-256 mismatch!\nExpected: {expected_sha}\nActual:   {actual_sha}"
    )

