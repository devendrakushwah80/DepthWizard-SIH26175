"""
DepthWizard (SIH26175) — Health & System Info Unit Tests
Player 4: Backend & Systems Integration Lead
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert data["model"]["loaded"] is True
    assert data["model"]["name"].startswith("M2-FINAL") or data["model"]["name"].startswith("M3-FINAL")
    assert data["model"]["checkpoint_sha256_verified"] is True
    assert data["model"]["checkpoint_sha256"] in [
        "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f",
        "db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330",
    ]
    assert data["model"]["output_parameterization"] == "softplus"
    assert data["model"]["dav2_frozen"] is True
    assert "gpu" in data
    assert "modules" in data
    assert data["modules"]["inference_service"] is True
    assert data["modules"]["geometry_3d_engine"] is True

def test_system_info_endpoint():
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "application_name" in data
    assert "version" in data
    assert "PNG" in data["supported_input_formats"]
    assert "GeoTIFF (.tif/.tiff)" in data["supported_input_formats"]
    assert data["default_mesh_resolution"] == 384
    assert 384 in data["available_mesh_resolutions"]
