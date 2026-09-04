"""Hermetic backend fixtures; no downloaded dataset or remote model is required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.app.config import settings
from backend.app.services.inference_service import inference_service
from backend.app.services.model_service import model_service
from backend.app.services.scene_service import scene_service


class _FrozenModelStub:
    @staticmethod
    def parameters():
        return ()


class _Dav2Stub:
    model = _FrozenModelStub()


def _deterministic_inference(rgb: np.ndarray, *args, **kwargs) -> dict:
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    agl = (0.25 + 8.0 * (xx / max(width - 1, 1)) * (yy / max(height - 1, 1))).astype(np.float32)
    return {
        "predicted_agl_m": agl,
        "dav2_prior": np.zeros_like(agl),
        "relative_surface": (agl / np.max(agl)).astype(np.float32),
        "dav2_time_s": 0.0,
        "m2_time_s": 0.0,
        "ai_time_s": 0.0,
        "total_inference_time_s": 0.0,
        "peak_vram_mb": 0.0,
        "window_count": 1,
        "window_size_px": 512,
        "window_step_px": 256,
        "blend": "test fixture",
    }


@pytest.fixture(scope="session", autouse=True)
def isolated_runtime(tmp_path_factory):
    runtime_root = tmp_path_factory.mktemp("depthwizard_runtime")
    scene_root = runtime_root / "scenes"
    upload_root = runtime_root / "uploads"
    scene_root.mkdir()
    upload_root.mkdir()

    original = {
        "storage": settings.STORAGE_DIR,
        "uploads": settings.UPLOAD_TEMP_DIR,
        "scene_storage": scene_service.storage_dir,
        "predict": inference_service.predict_height_map,
        "initialized": model_service._initialized,
        "device": model_service.device,
        "device_name": model_service.device_name,
        "dav2": model_service.dav2_model,
        "m2": model_service.m2_model,
        "sha": model_service.checkpoint_sha256,
    }
    settings.STORAGE_DIR = str(scene_root)
    settings.UPLOAD_TEMP_DIR = str(upload_root)
    scene_service.storage_dir = str(scene_root)
    inference_service.predict_height_map = _deterministic_inference
    model_service._initialized = True
    model_service.device = "cpu"
    model_service.device_name = "CPU test fixture"
    model_service.dav2_model = _Dav2Stub()
    model_service.m2_model = object()
    model_service.checkpoint_sha256 = settings.MODEL_CHECKPOINT_SHA256

    yy, xx = np.mgrid[:64, :64]
    rgb = np.stack([xx * 4, yy * 4, np.full_like(xx, 128)], axis=-1).astype(np.uint8)
    agl = (0.25 + 10.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 180.0)).astype(np.float32)
    scene_service.create_scene_products(
        scene_id="NYC_00735",
        rgb_image=rgb,
        predicted_agl=agl,
        relative_surface=(agl / np.max(agl)).astype(np.float32),
        spatial_meta={
            "format": "PNG",
            "is_georeferenced": False,
            "crs": None,
            "transform": None,
            "bounds": None,
            "gsd_m": 0.5,
            "gsd_source": "test fixture",
            "resolution": None,
        },
        mesh_resolution=32,
    )

    yield

    settings.STORAGE_DIR = original["storage"]
    settings.UPLOAD_TEMP_DIR = original["uploads"]
    scene_service.storage_dir = original["scene_storage"]
    inference_service.predict_height_map = original["predict"]
    model_service._initialized = original["initialized"]
    model_service.device = original["device"]
    model_service.device_name = original["device_name"]
    model_service.dav2_model = original["dav2"]
    model_service.m2_model = original["m2"]
    model_service.checkpoint_sha256 = original["sha"]
