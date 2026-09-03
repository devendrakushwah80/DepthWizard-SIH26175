"""
DepthWizard (SIH26175) — Scene Management & Asset Delivery Tests
Player 4: Backend & Systems Integration Lead
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_list_scenes():
    response = client.get("/api/v1/scenes")
    assert response.status_code == 200
    scenes = response.json()
    assert isinstance(scenes, list)
    assert len(scenes) >= 1
    scene_ids = [s["scene_id"] for s in scenes]
    assert "NYC_00735" in scene_ids

def test_get_scene_metadata():
    response = client.get("/api/v1/scenes/NYC_00735")
    assert response.status_code == 200
    data = response.json()
    assert data["scene_id"] == "NYC_00735"
    assert data["status"] == "completed"
    assert "spatial_info" in data
    assert data["spatial_info"]["gsd_m"] == 0.5
    assert "height_stats" in data
    assert data["height_stats"]["unit"] == "metres"
    assert "products" in data
    assert data["products"]["mesh_glb"] is True

def test_get_mesh_glb():
    response = client.get("/api/v1/scenes/NYC_00735/mesh.glb")
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"
    assert len(response.content) > 1000

def test_get_pointcloud_ply():
    response = client.get("/api/v1/scenes/NYC_00735/pointcloud.ply")
    assert response.status_code == 200
    assert len(response.content) > 1000

def test_get_textures():
    r_rgb = client.get("/api/v1/scenes/NYC_00735/texture/rgb")
    assert r_rgb.status_code == 200
    assert "image" in r_rgb.headers["content-type"]

    r_hgt = client.get("/api/v1/scenes/NYC_00735/heightmap")
    assert r_hgt.status_code == 200
    assert "image" in r_hgt.headers["content-type"]

    r_slope = client.get("/api/v1/scenes/NYC_00735/slope")
    assert r_slope.status_code == 200
    assert "image" in r_slope.headers["content-type"]

def test_nonexistent_scene():
    response = client.get("/api/v1/scenes/NON_EXISTENT_SCENE_123")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "SCENE_NOT_FOUND"
