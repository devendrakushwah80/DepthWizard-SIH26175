"""
DepthWizard (SIH26175) — System & Health Schemas
Player 4: Backend & Systems Integration Lead
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class ModelHealthInfo(BaseModel):
    loaded: bool = True
    name: str = "M3-FINAL + Frozen Depth Anything V2 Small"
    device: str = "cuda"
    checkpoint_path: str
    vram_allocated_mb: float = 0.0
    checkpoint_sha256: Optional[str] = None
    checkpoint_sha256_verified: bool = False
    output_parameterization: str = "softplus"
    dav2_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    dav2_frozen: bool = True

class GPUHealthInfo(BaseModel):
    available: bool = True
    device_name: str
    total_memory_mb: float
    allocated_memory_mb: float
    free_memory_mb: float

class ModulesHealthInfo(BaseModel):
    inference_service: bool = True
    geospatial_service: bool = True
    geometry_3d_engine: bool = True
    job_manager: bool = True

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model: ModelHealthInfo
    gpu: GPUHealthInfo
    modules: ModulesHealthInfo

class SystemInfoResponse(BaseModel):
    application_name: str
    version: str
    supported_input_formats: List[str] = ["PNG", "JPEG", "JPG", "GeoTIFF (.tif/.tiff)"]
    max_upload_size_mb: int
    default_mesh_resolution: int
    available_mesh_resolutions: List[int] = [256, 384, 512]
    gsd_m_per_px_default: Optional[float] = None
    gpu_enabled: bool
    gpu_device: str
    absolute_dsm_supported: bool = True
