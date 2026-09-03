"""
DepthWizard (SIH26175) — Health & System Info Routes
Player 4: Backend & Systems Integration Lead
"""

from fastapi import APIRouter
from backend.app.config import settings
from backend.app.schemas.system import (
    HealthResponse,
    SystemInfoResponse,
    ModelHealthInfo,
    GPUHealthInfo,
    ModulesHealthInfo
)
from backend.app.services.model_service import model_service

router = APIRouter(tags=["System & Health"])

@router.get("/health", response_model=HealthResponse, summary="System Health & Model Status")
async def get_health():
    """
    Returns server status, loaded resident PyTorch models, and GPU memory metrics.
    """
    info = model_service.get_health_info()
    gpu = info.get('gpu_info', {})
    
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        model=ModelHealthInfo(
            loaded=info['loaded'],
            name=info['name'],
            device=info['device'],
            checkpoint_path=info['checkpoint_path'],
            vram_allocated_mb=info['vram_allocated_mb'],
            checkpoint_sha256=info.get('checkpoint_sha256'),
            checkpoint_sha256_verified=info.get('checkpoint_sha256_verified', False),
            output_parameterization=info.get('output_parameterization', 'softplus'),
            dav2_model_id=info.get('dav2_model_id', settings.DAV2_MODEL_ID),
            dav2_frozen=info.get('dav2_frozen', False),
        ),
        gpu=GPUHealthInfo(
            available=gpu.get('available', False),
            device_name=gpu.get('device_name', 'Unknown'),
            total_memory_mb=gpu.get('total_memory_mb', 0.0),
            allocated_memory_mb=gpu.get('allocated_memory_mb', 0.0),
            free_memory_mb=gpu.get('free_memory_mb', 0.0)
        ),
        modules=ModulesHealthInfo(
            inference_service=True,
            geospatial_service=True,
            geometry_3d_engine=True,
            job_manager=True
        )
    )

@router.get("/api/v1/system/info", response_model=SystemInfoResponse, summary="Frontend System Capabilities")
async def get_system_info():
    """
    Provides frontend-friendly capabilities, formats, and parameter limits.
    """
    return SystemInfoResponse(
        application_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        supported_input_formats=["PNG", "JPEG", "JPG", "GeoTIFF (.tif/.tiff)"],
        max_upload_size_mb=settings.MAX_UPLOAD_SIZE_MB,
        default_mesh_resolution=settings.DEFAULT_MESH_RESOLUTION,
        available_mesh_resolutions=[256, 384, 512],
        gsd_m_per_px_default=settings.DEFAULT_GSD_M,
        gpu_enabled=model_service.device == 'cuda',
        gpu_device=model_service.device_name,
        absolute_dsm_supported=True
    )
