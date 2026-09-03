import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.services.geospatial_service import geospatial_service


def _write_tif(path, data, transform, crs="EPSG:32618"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as destination:
        destination.write(data.astype(np.float32), 1)


def test_aligned_dem_produces_absolute_dsm_identity(tmp_path):
    transform = from_origin(500000, 4500000, 0.5, 0.5)
    dem = np.arange(48, dtype=np.float32).reshape(6, 8) + 100
    agl = np.full((6, 8), 3.25, dtype=np.float32)
    dem_path = tmp_path / "aligned_dem.tif"
    _write_tif(dem_path, dem, transform)

    result = geospatial_service.align_and_compute_dsm(
        agl,
        str(dem_path),
        {
            "is_georeferenced": True,
            "crs": "EPSG:32618",
            "transform": list(transform),
        },
    )

    np.testing.assert_array_equal(result["base_dem"], dem)
    np.testing.assert_array_equal(result["absolute_dsm"], dem + agl)


def test_coarse_dem_is_resampled_onto_optical_grid(tmp_path):
    """Coarse DEM (e.g., 30m like SRTM) is resampled onto high-res optical grid."""
    # Optical tile: 0.5m GSD, 16x16 pixels -> 8m x 8m extent
    optical_transform = from_origin(500000, 4500000, 0.5, 0.5)
    # Coarse DEM: 30m resolution covering a larger area around the optical tile
    dem_transform = from_origin(499980, 4500020, 30.0, 30.0)
    coarse_dem = np.full((4, 4), 150.0, dtype=np.float32)
    dem_path = tmp_path / "coarse_dem.tif"
    _write_tif(dem_path, coarse_dem, dem_transform)

    agl = np.full((16, 16), 12.5, dtype=np.float32)
    result = geospatial_service.align_and_compute_dsm(
        agl,
        str(dem_path),
        {
            "is_georeferenced": True,
            "crs": "EPSG:32618",
            "transform": list(optical_transform),
        },
    )

    assert result["absolute_dsm_available"] is True
    assert result["base_dem"].shape == (16, 16)
    assert result["absolute_dsm"].shape == (16, 16)
    # Numeric identity: absolute_dsm - base_dem == agl
    np.testing.assert_allclose(result["absolute_dsm"] - result["base_dem"], agl, rtol=1e-5, atol=1e-5)
    assert result["dem_provenance"]["resampling_method"] == "bilinear"
    assert result["coverage_pct"] > 0


def test_non_overlapping_dem_is_rejected(tmp_path):
    """DEM that does not overlap the optical tile must be cleanly rejected."""
    optical_transform = from_origin(500000, 4500000, 0.5, 0.5)
    # Far-away DEM (e.g. 100km away)
    dem_path = tmp_path / "far_away_dem.tif"
    _write_tif(
        dem_path,
        np.ones((6, 8), dtype=np.float32),
        from_origin(600000, 4600000, 0.5, 0.5),
    )

    with pytest.raises(ValueError, match="does not spatially overlap"):
        geospatial_service.align_and_compute_dsm(
            np.ones((6, 8), dtype=np.float32),
            str(dem_path),
            {
                "is_georeferenced": True,
                "crs": "EPSG:32618",
                "transform": list(optical_transform),
            },
        )

