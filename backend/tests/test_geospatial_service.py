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


def test_misaligned_dem_is_rejected(tmp_path):
    optical_transform = from_origin(500000, 4500000, 0.5, 0.5)
    dem_path = tmp_path / "misaligned_dem.tif"
    _write_tif(
        dem_path,
        np.ones((6, 8), dtype=np.float32),
        from_origin(500001, 4500000, 0.5, 0.5),
    )

    with pytest.raises(ValueError, match="affine transform/origin differs"):
        geospatial_service.align_and_compute_dsm(
            np.ones((6, 8), dtype=np.float32),
            str(dem_path),
            {
                "is_georeferenced": True,
                "crs": "EPSG:32618",
                "transform": list(optical_transform),
            },
        )
