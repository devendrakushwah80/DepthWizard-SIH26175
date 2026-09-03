"""
DepthWizard (SIH26175) — Processing Job Routes
Player 4: Backend & Systems Integration Lead
"""

import os
import uuid
import shutil
import re
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.app.schemas.jobs import JobCreateResponse, JobStatusResponse
from backend.app.services.job_service import job_service
from backend.app.core.errors import JobNotFoundError, InvalidFileTypeError
from backend.app.config import settings

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs & Processing"])

@router.post("", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED, summary="Submit Image for 3D Reconstruction")
async def create_processing_job(
    file: UploadFile = File(..., description="Optical Satellite Image (PNG, JPG, JPEG, or GeoTIFF)"),
    dem_file: Optional[UploadFile] = File(None, description="Optional Base DEM GeoTIFF for Absolute DSM computation"),
    generate_3d: bool = Form(True, description="Generate physical 1x 3D surface mesh (.glb/.obj)"),
    generate_pointcloud: bool = Form(True, description="Generate dense 3D point cloud (.ply)"),
    mesh_resolution: int = Form(384, description="Mesh grid resolution (256, 384, or 512)"),
    vertical_exaggeration: float = Form(1.0, description="Vertical scaling factor (default 1.0 for physical mode)"),
    gsd_m: Optional[float] = Form(None, description="Optional user-supplied horizontal GSD for non-georeferenced RGB"),
    scene_name: Optional[str] = Form(None, description="Optional human-readable scene identifier")
):
    """
    Submits an optical image for asynchronous height estimation, geospatial alignment, and 3D surface reconstruction.
    Returns immediately with a job UUID.
    """
    filename = file.filename or "upload.png"
    ext = os.path.splitext(filename)[1].lower()
    
    allowed_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    if ext not in allowed_exts:
        raise InvalidFileTypeError(f"Unsupported file extension '{ext}'. Allowed: {', '.join(allowed_exts)}")

    is_geotiff = ext in ['.tif', '.tiff']

    if file.size is not None and (file.size <= 0 or file.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload must be between 1 byte and {settings.MAX_UPLOAD_SIZE_MB} MB",
        )
    if mesh_resolution not in (256, 384, 512):
        raise HTTPException(status_code=422, detail="mesh_resolution must be 256, 384, or 512")
    if abs(vertical_exaggeration - 1.0) > 1e-9:
        raise HTTPException(
            status_code=422,
            detail="Vertical exaggeration is display-only; generated artifacts must use 1.0x",
        )
    if gsd_m is not None and not (0.001 <= gsd_m <= 1000.0):
        raise HTTPException(status_code=422, detail="gsd_m must be between 0.001 and 1000 metres/pixel")
    if dem_file and not is_geotiff:
        raise HTTPException(
            status_code=422,
            detail="An aligned DEM requires a georeferenced GeoTIFF optical input",
        )
    if dem_file:
        dem_ext = os.path.splitext(dem_file.filename or "")[1].lower()
        if dem_ext not in ('.tif', '.tiff'):
            raise InvalidFileTypeError("Base DEM must be a GeoTIFF (.tif/.tiff)")
    if scene_name:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", scene_name):
            raise HTTPException(
                status_code=422,
                detail="scene_name must be 1-64 characters using letters, numbers, '_' or '-'",
            )
        if os.path.exists(os.path.join(settings.STORAGE_DIR, scene_name)):
            raise HTTPException(status_code=409, detail=f"Scene '{scene_name}' already exists")

    # Create Job
    job_id = job_service.create_job()
    
    # Save Upload to Temp Storage
    safe_filename = os.path.basename(filename)
    temp_upload_path = os.path.join(settings.UPLOAD_TEMP_DIR, f"{job_id}_{safe_filename}")
    with open(temp_upload_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Save DEM if provided
    temp_dem_path = None
    if dem_file:
        safe_dem_filename = os.path.basename(dem_file.filename or "base_dem.tif")
        temp_dem_path = os.path.join(settings.UPLOAD_TEMP_DIR, f"{job_id}_dem_{safe_dem_filename}")
        with open(temp_dem_path, 'wb') as f:
            shutil.copyfileobj(dem_file.file, f)

    # Launch Pipeline Asynchronously
    job_service.run_pipeline_async(
        job_id=job_id,
        file_path=temp_upload_path,
        is_geotiff=is_geotiff,
        dem_file_path=temp_dem_path,
        user_gsd_m=gsd_m,
        generate_3d=generate_3d,
        generate_pointcloud=generate_pointcloud,
        mesh_resolution=mesh_resolution,
        vertical_exaggeration=vertical_exaggeration,
        custom_scene_id=scene_name
    )

    job_info = job_service.get_job_status(job_id)
    return JobCreateResponse(
        job_id=job_id,
        status=job_info['status'],
        created_at=job_info['created_at'],
        message="Image accepted for M2-FINAL asynchronous processing"
    )

@router.get("/{job_id}", response_model=JobStatusResponse, summary="Query Job Processing Status")
async def get_job_status(job_id: str):
    """
    Returns the real-time stage progress, timestamps, and completed scene_id.
    """
    job_info = job_service.get_job_status(job_id)
    if not job_info:
        raise JobNotFoundError(job_id)
        
    return JobStatusResponse(**job_info)
