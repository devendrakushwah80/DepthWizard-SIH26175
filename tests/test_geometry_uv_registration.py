import numpy as np
import trimesh

from src.geometry.gltf_export import export_mesh_glb
from src.geometry.raster_to_mesh import build_terrain_mesh
from src.geometry.scene import Scene3D


def test_glb_round_trip_preserves_mesh_image_row_mapping(tmp_path):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[0, -1] = [0, 255, 0]
    rgb[-1, 0] = [0, 0, 255]
    rgb[-1, -1] = [255, 255, 0]
    agl = np.arange(6, dtype=np.float32).reshape(2, 3)
    scene = Scene3D("uv_test", rgb, agl)
    mesh = build_terrain_mesh(scene, mesh_resolution=2, add_side_skirts=False)
    output = tmp_path / "uv_test.glb"

    export_mesh_glb(mesh, str(output))
    loaded = trimesh.load(output, force="scene", process=False)
    geometry = next(iter(loaded.geometry.values()))

    # Trimesh exposes loaded glTF UVs in its bottom-left internal convention;
    # row 0 therefore round-trips to V=1.
    np.testing.assert_allclose(
        geometry.visual.uv[:4],
        np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32),
        atol=1e-7,
    )


def test_mesh_height_rows_share_raster_row_order():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    agl = np.arange(12, dtype=np.float32).reshape(3, 4)
    scene = Scene3D("row_test", rgb, agl)
    mesh = build_terrain_mesh(
        scene,
        mesh_resolution=3,
        downsample_method="bilinear",
        add_side_skirts=False,
    )
    surface = mesh.vertices[:9].reshape(3, 3, 3)

    assert surface[0, 0, 2] > surface[-1, 0, 2]
    assert surface[0, :, 1].mean() < surface[-1, :, 1].mean()
    np.testing.assert_allclose(mesh.uvs[:3, 1], 1.0)
    np.testing.assert_allclose(mesh.uvs[6:9, 1], 0.0)


def test_full_resolution_surface_preserves_every_raster_sample_and_uv_corner():
    height, width = 7, 7
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    agl = np.arange(height * width, dtype=np.float32).reshape(height, width)
    scene = Scene3D("full_grid_registration", rgb, agl, gsd=0.5)

    mesh = build_terrain_mesh(
        scene,
        mesh_resolution=max(height, width),
        downsample_method="bilinear",
        add_side_skirts=False,
    )
    surface = mesh.vertices[: height * width].reshape(height, width, 3)
    uvs = mesh.uvs[: height * width].reshape(height, width, 2)

    np.testing.assert_array_equal(surface[:, :, 1], agl)
    np.testing.assert_allclose(surface[0, :, 2], height * 0.5 / 2)
    np.testing.assert_allclose(surface[-1, :, 2], -height * 0.5 / 2)
    np.testing.assert_allclose(surface[:, 0, 0], -width * 0.5 / 2)
    np.testing.assert_allclose(surface[:, -1, 0], width * 0.5 / 2)
    np.testing.assert_allclose(uvs[0, 0], [0, 1])
    np.testing.assert_allclose(uvs[0, -1], [1, 1])
    np.testing.assert_allclose(uvs[-1, 0], [0, 0])
    np.testing.assert_allclose(uvs[-1, -1], [1, 0])
