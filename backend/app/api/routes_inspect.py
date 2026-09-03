"""
DepthWizard (SIH26175) — Inspection & Measurement Routes
Player 4: Backend & Systems Integration Lead
"""

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.inspection import InspectRequest, InspectResponse, MeasureRequest, MeasureResponse
from backend.app.services.scene_service import scene_service
from backend.app.core.errors import SceneNotFoundError

router = APIRouter(prefix="/api/v1/scenes/{scene_id}", tags=["Inspection & Spatial Analysis"])

@router.post("/inspect", response_model=InspectResponse, summary="Authoritative Full-Resolution Pixel Inspection")
async def inspect_pixel(scene_id: str, payload: InspectRequest):
    """
    Queries the authoritative, uncompressed full-resolution 1024x1024 float32 raster.
    Does NOT query the downsampled 384x384 3D web mesh for physical height values.
    """
    if not scene_service.scene_exists(scene_id):
        raise SceneNotFoundError(scene_id)
        
    res = scene_service.inspect_pixel(scene_id, payload.u, payload.v)
    if not res:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to sample raster array")
        
    return InspectResponse(**res)

@router.post("/measure", response_model=MeasureResponse, summary="3D Spatial Distance & Slope Measurement")
async def measure_distance(scene_id: str, payload: MeasureRequest):
    """
    Calculates 3D Euclidean distance, horizontal baseline, vertical elevation delta Δh, and slope.
    """
    if not scene_service.scene_exists(scene_id):
        raise SceneNotFoundError(scene_id)
        
    try:
        res = scene_service.measure_distance(scene_id, payload.point_a, payload.point_b, is_pixel=payload.is_pixel_coords)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if not res:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate 3D measurement")
        
    return MeasureResponse(**res)
