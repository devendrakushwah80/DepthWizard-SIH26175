"""
DepthWizard (SIH26175) — Geospatial Exit Gate Regression Tests
Tests EPSG:4326 geodesic handling, cross-CRS DEM alignment, NoData holes, and coarse SRTM workflow.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from backend.app.schemas.jobs import JobStatus
from backend.app.services.geospatial_service import GeospatialService
from backend.app.services.job_service import job_service
from backend.app.services.scene_service import scene_service


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_optical_geotiff(
    path: Path,
    width: int = 64,
    height: int = 64,
    crs: str = "EPSG:32618",
    origin_x: float = 500000.0,
    origin_y: float = 4500000.0,
    res_x: float = 0.5,
    res_y: float = 0.5,
) -> Path:
    transform = from_origin(origin_x, origin_y, res_x, res_y)
    yy, xx = np.mgrid[:height, :width]
    data = np.stack([xx * 3, yy * 3, np.full_like(xx, 140)], axis=0).astype(np.uint8)
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
    res_x: float = 0.5,
    res_y: float = 0.5,
    base_elev: float = 30.0,
    nodata_val: float = -9999.0,
    nodata_mask: np.ndarray | None = None,
) -> Path:
    transform = from_origin(origin_x, origin_y, res_x, res_y)
    yy, xx = np.mgrid[:height, :width]
    data = (base_elev + 0.1 * xx + 0.05 * yy).astype(np.float32)
    if nodata_mask is not None:
        data[nodata_mask] = nodata_val

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
        nodata=nodata_val,
    ) as dst:
        dst.write(data, 1)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_case_a_epsg4326_optical_geotiff(tmp_path):
    """Exit Gate 1.A: EPSG:4326 optical GeoTIFF.
    Verify:
    - Original CRS remains EPSG:4326.
    - Degree units are never interpreted directly as metres.
    - Horizontal GSD uses geodesic WGS-84 calculation at scene latitude.
    - Metric slope calculation is metric-correct.
    - Output rasters preserve original CRS and grid.
    """
    # Create optical raster at lat ~28.6139, lon ~77.2090 (New Delhi)
    # Resolution = 0.000005 deg (~0.55m latitude, ~0.49m longitude)
    res_deg = 0.000005
    origin_lon = 77.2090
    origin_lat = 28.6139
    tif_path = _create_optical_geotiff(
        tmp_path / "delhi_4326.tif",
        width=64,
        height=64,
        crs="EPSG:4326",
        origin_x=origin_lon,
        origin_y=origin_lat,
        res_x=res_deg,
        res_y=res_deg,
    )

    # 1. Test parsing via GeospatialService directly
    geo_svc = GeospatialService()
    meta = geo_svc.parse_geotiff(str(tif_path))

    assert meta["is_georeferenced"] is True
    assert meta["crs"] == "EPSG:4326"
    # Native resolution must remain in degrees (e.g. 5e-6)
    assert meta["resolution"][0] == pytest.approx(res_deg)
    assert meta["resolution"][1] == pytest.approx(res_deg)

    # GSD must NOT be 0.000005 metres! It must be geodesic ground distance (~0.45m to 0.60m)
    assert meta["gsd_m"] is not None
    assert 0.40 <= meta["gsd_m"] <= 0.65
    assert meta["gsd_source"] == "geodesic_wgs84"
    assert meta["horizontal_units"] == "metres"

    # 2. Test full end-to-end scene processing pipeline
    job_id = job_service.create_job()
    scene_id = f"test_4326_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(tif_path),
            is_geotiff=True,
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

        scene_meta = scene_service.load_scene_metadata(scene_id)
        assert scene_meta is not None
        spatial = scene_meta["spatial_info"]
        assert spatial["crs"] == "EPSG:4326"
        assert spatial["gsd_source"] == "geodesic_wgs84"
        assert 0.40 <= spatial["gsd_m"] <= 0.65

        # Check exported GeoTIFF preserves EPSG:4326 and affine transform
        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        agl_tif = scene_dir / "rasters" / "predicted_agl.tif"
        assert agl_tif.exists()
        with rasterio.open(agl_tif) as src:
            assert src.crs.to_string() == "EPSG:4326"
            assert src.transform.a == pytest.approx(res_deg)
            assert src.transform.e == pytest.approx(-res_deg)
            assert src.transform.c == pytest.approx(origin_lon)
            assert src.transform.f == pytest.approx(origin_lat)
            assert src.shape == (64, 64)

        # Inspect pixel: slope and world coordinates must be metric-correct
        insp = scene_service.inspect_pixel(scene_id, 32, 32)
        assert insp["world_coordinates_m"] is not None
        # World coordinates in metres must be on the scale of tens of metres (64 * ~0.5m = ~32m extent)
        assert abs(insp["world_coordinates_m"]["x"]) < 40.0
        assert abs(insp["world_coordinates_m"]["z"]) < 40.0

        # Distance measurement across 10 pixels: ~10 * 0.52m = ~5.2m
        meas = scene_service.measure_distance(scene_id, [20, 20], [30, 20], is_pixel=True)
        assert meas is not None
        assert 4.0 <= meas["horizontal_distance_m"] <= 6.5
        assert 0.0 <= meas["slope_angle_deg"] <= 90.0

    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_b_cross_crs_dem_reprojection(tmp_path):
    """Exit Gate 1.B: Optical GeoTIFF + DEM in DIFFERENT CRS.
    Optical: UTM Zone 18N (EPSG:32618).
    DEM: Geographic WGS-84 (EPSG:4326).
    Verify:
    - DEM is reprojected geospatially from EPSG:4326 to EPSG:32618.
    - DEM is resampled to exact optical grid.
    - Output CRS/transform/shape exactly match optical raster.
    - Absolute DSM is generated.
    - Identity: absolute_dsm - aligned_dem == predicted_agl.
    """
    # Optical in UTM 18N (metres)
    opt_x = 583000.0
    opt_y = 4510000.0
    opt_res = 0.5
    width, height = 64, 64
    opt_tif = _create_optical_geotiff(
        tmp_path / "optical_utm.tif",
        width=width,
        height=height,
        crs="EPSG:32618",
        origin_x=opt_x,
        origin_y=opt_y,
        res_x=opt_res,
        res_y=opt_res,
    )

    # Compute optical bounds in EPSG:4326 to place the DEM accurately
    opt_bounds_wgs84 = transform_bounds(
        "EPSG:32618",
        "EPSG:4326",
        opt_x,
        opt_y - height * opt_res,
        opt_x + width * opt_res,
        opt_y,
    )
    # Add 0.01 deg padding around the optical area for the DEM
    dem_left = opt_bounds_wgs84[0] - 0.005
    dem_bottom = opt_bounds_wgs84[1] - 0.005
    dem_right = opt_bounds_wgs84[2] + 0.005
    dem_top = opt_bounds_wgs84[3] + 0.005
    dem_res_deg = 0.0005  # ~50m resolution
    dem_w = int(np.ceil((dem_right - dem_left) / dem_res_deg))
    dem_h = int(np.ceil((dem_top - dem_bottom) / dem_res_deg))

    dem_tif = _create_dem_geotiff(
        tmp_path / "dem_wgs84.tif",
        width=dem_w,
        height=dem_h,
        crs="EPSG:4326",
        origin_x=dem_left,
        origin_y=dem_top,
        res_x=dem_res_deg,
        res_y=dem_res_deg,
        base_elev=120.0,
    )

    job_id = job_service.create_job()
    scene_id = f"test_cross_crs_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(opt_tif),
            is_geotiff=True,
            dem_file_path=str(dem_tif),
            user_gsd_m=None,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED

        scene_meta = scene_service.load_scene_metadata(scene_id)
        assert scene_meta is not None
        prods = scene_meta["products"]
        assert prods["absolute_dsm_tif"] is True
        assert prods["aligned_terrain_dem_tif"] is True

        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        agl_path = scene_dir / "rasters" / "predicted_agl.tif"
        dem_path = scene_dir / "rasters" / "aligned_terrain_dem.tif"
        dsm_path = scene_dir / "rasters" / "absolute_dsm.tif"

        # Verify output CRS, transform, and shape match optical grid
        with rasterio.open(agl_path) as s_agl, rasterio.open(dem_path) as s_dem, rasterio.open(dsm_path) as s_dsm:
            assert s_dem.crs.to_string() == "EPSG:32618"
            assert s_dsm.crs.to_string() == "EPSG:32618"
            assert s_dem.shape == (height, width)
            assert s_dsm.shape == (height, width)
            assert s_dem.transform == s_agl.transform
            assert s_dsm.transform == s_agl.transform

            agl_arr = s_agl.read(1)
            dem_arr = s_dem.read(1)
            dsm_arr = s_dsm.read(1)

        # Verify numerical identity
        np.testing.assert_allclose(dsm_arr - dem_arr, agl_arr, atol=1e-4)

        # Verify provenance
        prov = scene_meta["dem_provenance"]
        assert prov["original_dem_crs"] == "EPSG:4326"
        assert prov["coverage_percentage"] >= 99.0
        assert prov["resampling_method"] == "bilinear"

    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_case_c_dem_with_nodata_holes(tmp_path):
    """Exit Gate 1.C: DEM with NoData holes.
    Verify:
    - NoData is explicitly detected.
    - Invalid areas are masked consistently as NaN in aligned DEM and absolute DSM.
    - No fabricated terrain elevations are inserted into holes.
    - Metadata records nodata statistics (pixels and percentage).
    """
    width, height = 64, 64
    opt_tif = _create_optical_geotiff(
        tmp_path / "opt_nodata.tif",
        width=width,
        height=height,
        crs="EPSG:32618",
        origin_x=500000.0,
        origin_y=4500000.0,
        res_x=0.5,
        res_y=0.5,
    )

    # Create a 16x16 hole in the center of the DEM
    nodata_mask = np.zeros((height, width), dtype=bool)
    nodata_mask[24:40, 24:40] = True
    hole_pixel_count = int(np.count_nonzero(nodata_mask))

    dem_tif = _create_dem_geotiff(
        tmp_path / "dem_with_hole.tif",
        width=width,
        height=height,
        crs="EPSG:32618",
        origin_x=500000.0,
        origin_y=4500000.0,
        res_x=0.5,
        res_y=0.5,
        base_elev=50.0,
        nodata_val=-9999.0,
        nodata_mask=nodata_mask,
    )

    job_id = job_service.create_job()
    scene_id = f"test_nodata_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(opt_tif),
            is_geotiff=True,
            dem_file_path=str(dem_tif),
            user_gsd_m=None,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED

        scene_meta = scene_service.load_scene_metadata(scene_id)
        assert scene_meta is not None
        prov = scene_meta["dem_provenance"]
        assert prov is not None

        # Verify nodata statistics in metadata
        assert prov["nodata_pixels"] == hole_pixel_count
        expected_nodata_pct = (hole_pixel_count / (width * height)) * 100.0
        assert prov["nodata_percentage"] == pytest.approx(expected_nodata_pct, abs=0.1)

        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        aligned_dem_arr = np.load(scene_dir / "rasters" / "base_dem.npy")
        abs_dsm_arr = np.load(scene_dir / "rasters" / "absolute_dsm.npy")
        agl_arr = np.load(scene_dir / "rasters" / "predicted_agl.npy")

        # In the hole, DEM and DSM must be NaN (NO fabricated elevations inserted)
        assert np.all(np.isnan(aligned_dem_arr[24:40, 24:40]))
        assert np.all(np.isnan(abs_dsm_arr[24:40, 24:40]))

        # Outside the hole, values must be finite and identity must hold
        valid_mask = ~nodata_mask
        assert np.all(np.isfinite(aligned_dem_arr[valid_mask]))
        assert np.all(np.isfinite(abs_dsm_arr[valid_mask]))
        np.testing.assert_allclose(
            abs_dsm_arr[valid_mask] - aligned_dem_arr[valid_mask],
            agl_arr[valid_mask],
            atol=1e-4,
        )

    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)


def test_coarse_srtm_workflow(tmp_path):
    """Exit Gate 2: Coarse SRTM Workflow.
    High-res optical raster (0.5m, 64x64, UTM Zone 18N).
    Coarse DEM resembling SRTM 30m resolution (30m, 8x8, Geographic EPSG:4326).
    Verify and report:
    - original DEM resolution
    - aligned DEM resolution
    - original CRS
    - target CRS
    - coverage %
    - resampling method
    - valid/nodata pixel counts
    - identity: absolute_dsm - aligned_dem == predicted_agl
    - max absolute error, mean absolute error, valid pixel count
    """
    opt_x = 500000.0
    opt_y = 4500000.0
    opt_res = 0.5
    width, height = 64, 64
    opt_tif = _create_optical_geotiff(
        tmp_path / "srtm_opt.tif",
        width=width,
        height=height,
        crs="EPSG:32618",
        origin_x=opt_x,
        origin_y=opt_y,
        res_x=opt_res,
        res_y=opt_res,
    )

    # Convert optical bounds to EPSG:4326
    opt_bounds_wgs84 = transform_bounds(
        "EPSG:32618",
        "EPSG:4326",
        opt_x,
        opt_y - height * opt_res,
        opt_x + width * opt_res,
        opt_y,
    )
    # 30m in degrees is ~0.0002777 deg
    srtm_res_deg = 30.0 / 111320.0  # approx 0.00027 deg
    dem_left = opt_bounds_wgs84[0] - 3.0 * srtm_res_deg
    dem_top = opt_bounds_wgs84[3] + 3.0 * srtm_res_deg

    coarse_dem = _create_dem_geotiff(
        tmp_path / "srtm_coarse_30m.tif",
        width=10,
        height=10,
        crs="EPSG:4326",
        origin_x=dem_left,
        origin_y=dem_top,
        res_x=srtm_res_deg,
        res_y=srtm_res_deg,
        base_elev=250.0,
    )

    job_id = job_service.create_job()
    scene_id = f"test_srtm_wf_{uuid.uuid4().hex[:8]}"

    try:
        job_service._execute_pipeline(
            job_id=job_id,
            file_path=str(opt_tif),
            is_geotiff=True,
            dem_file_path=str(coarse_dem),
            user_gsd_m=None,
            generate_3d=True,
            generate_pointcloud=False,
            mesh_resolution=32,
            vertical_exaggeration=1.0,
            custom_scene_id=scene_id,
        )

        job = job_service.get_job_status(job_id)
        assert job["status"] == JobStatus.COMPLETED

        scene_meta = scene_service.load_scene_metadata(scene_id)
        assert scene_meta is not None
        prov = scene_meta["dem_provenance"]
        assert prov is not None

        # Verify reported metadata
        assert prov["original_dem_crs"] == "EPSG:4326"
        assert prov["coverage_percentage"] >= 99.0
        assert prov["resampling_method"] == "bilinear"
        assert prov["aligned_dem_resolution"] == [opt_res, opt_res]
        assert prov["nodata_pixels"] == 0

        scene_dir = Path(scene_service.get_scene_dir(scene_id))
        agl_arr = np.load(scene_dir / "rasters" / "predicted_agl.npy")
        dem_arr = np.load(scene_dir / "rasters" / "base_dem.npy")
        dsm_arr = np.load(scene_dir / "rasters" / "absolute_dsm.npy")

        valid_mask = np.isfinite(dsm_arr) & np.isfinite(dem_arr) & np.isfinite(agl_arr)
        valid_pixel_count = int(np.count_nonzero(valid_mask))
        assert valid_pixel_count == width * height

        diff = np.abs((dsm_arr[valid_mask] - dem_arr[valid_mask]) - agl_arr[valid_mask])
        max_abs_error = float(np.max(diff))
        mean_abs_error = float(np.mean(diff))

        assert max_abs_error < 1e-4
        assert mean_abs_error < 1e-5

        print(f"\n[SRTM Verification] Valid Pixels: {valid_pixel_count}")
        print(f"[SRTM Verification] Max Abs Error: {max_abs_error:.6e}")
        print(f"[SRTM Verification] Mean Abs Error: {mean_abs_error:.6e}")

    finally:
        shutil.rmtree(scene_service.get_scene_dir(scene_id), ignore_errors=True)
