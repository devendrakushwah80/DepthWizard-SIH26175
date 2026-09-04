"""
DepthWizard (SIH26175) — Local-First Acceptance Gate Regression Tests

Validates:
1. Default local route & configuration consistency (M3-FINAL)
2. Launcher M3 verification and selection logic
3. Full FastAPI REST contract (/health, /scenes, /jobs, inspection, measurement, artifacts)
4. Zero deployment-env dependency for localhost
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, settings
from backend.app.main import app
from scripts.start_depthwizard import verify_sha256, EXPECTED_M3_SHA256, EXPECTED_M2_SHA256

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m3_configuration_consistency():
    """Verify backend settings resolve to M3-FINAL consistently by default."""
    cfg = Settings()
    assert cfg.MODEL_FAMILY == "M3-FINAL"
    assert cfg.MODEL_NAME == "M3-FINAL"
    assert "M3_FINAL.pth" in str(cfg.MODEL_CHECKPOINT)
    assert cfg.MODEL_CHECKPOINT_SHA256 == "db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330"
    assert Path(cfg.MODEL_CHECKPOINT).exists(), f"M3 Checkpoint missing at {cfg.MODEL_CHECKPOINT}"


def test_launcher_m3_selection():
    """Verify launcher SHA verification functions on both M2 and M3 checkpoints."""
    m3_path = REPO_ROOT / "models" / "m3_final" / "M3_FINAL.pth"
    assert m3_path.exists()
    assert verify_sha256(m3_path, EXPECTED_M3_SHA256) is True

    m2_path = REPO_ROOT / "models" / "m2_final" / "M2_FINAL.pth"
    assert m2_path.exists()
    assert verify_sha256(m2_path, EXPECTED_M2_SHA256) is True


def test_no_deployment_env_required_for_localhost(monkeypatch):
    """Verify system config does not rely on GCP, Vercel, or Hugging Face Space envs."""
    for var in ["SPACE_ID", "VERCEL", "K_SERVICE", "GCP_PROJECT", "HF_TOKEN"]:
        monkeypatch.delenv(var, raising=False)

    clean_cfg = Settings()
    assert clean_cfg.MODEL_FAMILY == "M3-FINAL"
    assert Path(clean_cfg.STORAGE_DIR).exists()


def test_fastapi_rest_contract():
    """Verify the complete FastAPI REST contract for local workspace operation."""
    with TestClient(app) as client:
        # 1. Health
        res = client.get("/health")
        assert res.status_code == 200
        health = res.json()
        assert health["status"] == "ok"
        assert "M3-FINAL" in health["model"]["name"]

        # 2. System info
        res = client.get("/api/v1/system/info")
        assert res.status_code == 200

        # 3. Scene listing
        res = client.get("/api/v1/scenes")
        assert res.status_code == 200
        scenes = res.json()
        assert isinstance(scenes, list)
        scene_ids = [s["scene_id"] for s in scenes]
        assert "NYC_00735" in scene_ids

        # 4. Scene metadata
        res = client.get("/api/v1/scenes/NYC_00735")
        assert res.status_code == 200
        meta = res.json()
        assert meta["scene_id"] == "NYC_00735"
        assert "spatial_info" in meta
        assert "height_stats" in meta

        # 5. Artifact downloads / streams
        res = client.get("/api/v1/scenes/NYC_00735/mesh.glb")
        assert res.status_code == 200
        assert len(res.content) > 1000

        res = client.get("/api/v1/scenes/NYC_00735/texture/rgb")
        assert res.status_code == 200

        res = client.get("/api/v1/scenes/NYC_00735/heightmap")
        assert res.status_code == 200

        res = client.get("/api/v1/scenes/NYC_00735/relative_surface")
        assert res.status_code == 200

        res = client.get("/api/v1/scenes/NYC_00735/slope")
        assert res.status_code == 200

        # 6. Raycast inspection
        res = client.post("/api/v1/scenes/NYC_00735/inspect", json={"u": 512, "v": 512})
        assert res.status_code == 200
        inspect_data = res.json()
        assert "predicted_agl_m" in inspect_data
        assert "slope_deg" in inspect_data

        # 7. Measurement
        res = client.post(
            "/api/v1/scenes/NYC_00735/measure",
            json={"point_a": [100.0, 100.0], "point_b": [200.0, 200.0], "is_pixel_coords": True}
        )
        assert res.status_code == 200
        measure_data = res.json()
        assert "direct_3d_distance_m" in measure_data
        assert measure_data["direct_3d_distance_m"] > 0
