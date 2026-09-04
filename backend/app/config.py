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
    # Model family and checkpoints
    MODEL_FAMILY: str = os.getenv("DEPTHWIZARD_MODEL_FAMILY", "M3-FINAL")
    
    M2_MODEL_CHECKPOINT: str = _repository_path(
        "DEPTHWIZARD_M2_CHECKPOINT", BASE_DIR / "models/m2_final/M2_FINAL.pth"
    )
    M2_MODEL_CHECKPOINT_SHA256: str = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"

    M3_MODEL_CHECKPOINT: str = _repository_path(
        "DEPTHWIZARD_M3_CHECKPOINT", BASE_DIR / "models/m3_final/M3_FINAL.pth"
    )
    M3_MODEL_CHECKPOINT_SHA256: str = "db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330"

    # Active model configuration
    if MODEL_FAMILY == "M2-FINAL":
        MODEL_NAME: str = "M2-FINAL"
        MODEL_CHECKPOINT_SHA256: str = M2_MODEL_CHECKPOINT_SHA256
        _default_ckpt = BASE_DIR / "models/m2_final/M2_FINAL.pth"
    else:
        MODEL_NAME: str = "M3-FINAL"
        MODEL_CHECKPOINT_SHA256: str = M3_MODEL_CHECKPOINT_SHA256
        _default_ckpt = BASE_DIR / "models/m3_final/M3_FINAL.pth"

    _configured_ckpt = os.getenv("DEPTHWIZARD_MODEL_CHECKPOINT")
    if _configured_ckpt and (("m2" in _configured_ckpt.lower() and MODEL_NAME == "M3-FINAL") or ("m3" in _configured_ckpt.lower() and MODEL_NAME == "M2-FINAL")):
        MODEL_CHECKPOINT: str = _repository_path("DEPTHWIZARD_FAMILY_CHECKPOINT", _default_ckpt)
    else:
        MODEL_CHECKPOINT: str = _repository_path("DEPTHWIZARD_MODEL_CHECKPOINT", _default_ckpt)

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
    
    # CORS: environment-driven with local fallbacks
    _cors_env: str = os.getenv("DEPTHWIZARD_CORS_ORIGINS", "")
    _env_origins: list = [o.strip() for o in _cors_env.split(",") if o.strip()]
    
    CORS_ORIGINS: list = list(dict.fromkeys([
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "https://depth-wizard-sih-26175.vercel.app"
    ] + _env_origins))

settings = Settings()

# Ensure storage directories exist safely
try:
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_TEMP_DIR, exist_ok=True)
except Exception as e:
    print(f"[Config] Warning: Could not create storage directory: {e}")

# Seed default demo scene (NYC_00735) into storage directory if absent
try:
    import shutil
    dest_nyc = Path(settings.STORAGE_DIR) / "NYC_00735"
    if not dest_nyc.exists():
        seed_nyc = settings.BASE_DIR / "seed_scenes" / "NYC_00735"
        local_nyc = settings.BASE_DIR / "outputs" / "scenes" / "NYC_00735"
        source_nyc = seed_nyc if seed_nyc.exists() else (local_nyc if local_nyc.exists() else None)
        if source_nyc and source_nyc.resolve() != dest_nyc.resolve():
            print(f"[Config] Seeding demo scene NYC_00735 from {source_nyc} to {dest_nyc}...")
            shutil.copytree(source_nyc, dest_nyc, dirs_exist_ok=True)
            print("[Config] Demo scene NYC_00735 seeded successfully.")
except Exception as e:
    print(f"[Config] Notice: Seed scene check: {e}")

