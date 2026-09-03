"""
DepthWizard (SIH26175) — Inspection & Measurement Unit Tests
Player 4: Backend & Systems Integration Lead
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_inspect_pixel():
    payload = {"u": 32, "v": 32}
    response = client.post("/api/v1/scenes/NYC_00735/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scene_id"] == "NYC_00735"
    assert data["pixel"] == [32, 32]
    assert "predicted_agl_m" in data
    assert isinstance(data["predicted_agl_m"], float)
    assert data["predicted_agl_m"] >= 0.0
    assert "semantic_class" in data
    assert "world_coordinates_m" in data
    assert "x" in data["world_coordinates_m"]
    assert "z" in data["world_coordinates_m"]

def test_3d_spatial_measurement():
    payload = {
        "point_a": [10, 10],
        "point_b": [50, 50],
        "is_pixel_coords": True
    }
    response = client.post("/api/v1/scenes/NYC_00735/measure", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scene_id"] == "NYC_00735"
    assert "horizontal_distance_m" in data
    assert "direct_3d_distance_m" in data
    assert "elevation_delta_m" in data
    assert "slope_angle_deg" in data
    assert data["horizontal_distance_m"] > 0.0
    assert data["direct_3d_distance_m"] >= data["horizontal_distance_m"]
