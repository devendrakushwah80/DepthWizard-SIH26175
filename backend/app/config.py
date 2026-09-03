"""
DepthWizard (SIH26175) — Backend Configuration
Player 4: Backend & Systems Integration Lead
"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _repository_path(environment_name: str, default: Path) -> str:
    configured = os.getenv(environment_name)
    path = Path(configured).expanduser() if configured else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())

class Settings:
    PROJECT_NAME: str = "DepthWizard SIH26175 API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    MODEL_NAME: str = "M2-FINAL"
    MODEL_CHECKPOINT: str = _repository_path(
        "DEPTHWIZARD_MODEL_CHECKPOINT", BASE_DIR / "models/m2_final/M2_FINAL.pth"
    )
    MODEL_CHECKPOINT_SHA256: str = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"
    MODEL_OUTPUT_PARAMETERIZATION: str = "softplus"
    DAV2_MODEL_ID: str = os.getenv(
        "DEPTHWIZARD_DAV2_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf"
    )
    STORAGE_DIR: str = _repository_path(
        "DEPTHWIZARD_STORAGE_DIR", BASE_DIR / "outputs/scenes"
    )
    UPLOAD_TEMP_DIR: str = _repository_path(
        "DEPTHWIZARD_UPLOAD_TEMP_DIR", BASE_DIR / "outputs/temp_uploads"
    )
    
    # Model & Execution
    DEVICE: str = os.getenv("DEPTHWIZARD_DEVICE", os.getenv("DEVICE", "cuda")).lower()
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("DEPTHWIZARD_MAX_UPLOAD_SIZE_MB", "100"))
    DEFAULT_MESH_RESOLUTION: int = 384
    # A non-georeferenced image has no defensible horizontal metric scale unless
    # the user explicitly supplies one. Never substitute a project default.
    DEFAULT_GSD_M = None
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080"
    ]

settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_TEMP_DIR, exist_ok=True)
