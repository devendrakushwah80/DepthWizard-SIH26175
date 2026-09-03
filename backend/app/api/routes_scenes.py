"""
DepthWizard (SIH26175) — Scene & Artifact Delivery Routes
Player 4: Backend & Systems Integration Lead
"""

import os
from typing import List
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from backend.app.schemas.scenes import SceneMetadataResponse, SceneListItem
from backend.app.services.scene_service import scene_service
from backend.app.core.errors import SceneNotFoundError, AssetNotAvailableError
from backend.app.config import settings

router = APIRouter(prefix="/api/v1/scenes", tags=["Scenes & 3D Assets"])

@router.get("", response_model=List[SceneListItem], summary="List Processed 3D Scenes")
async def list_scenes():
    """
    Returns a catalog of all available reconstructed 3D scenes.
    """
    results = []
    if not os.path.exists(settings.STORAGE_DIR):
        return results

    for sid in os.listdir(settings.STORAGE_DIR):
        meta = scene_service.load_scene_metadata(sid)
        if meta:
            results.append(SceneListItem(
                scene_id=sid,
                created_at=meta.get('created_at', ''),
                input_format=meta.get('input_info', {}).get('format', 'RGB'),
                is_georeferenced=meta.get('spatial_info', {}).get('is_georeferenced', False),
                max_height_m=meta.get('height_stats', {}).get('max_m', 0.0),
                mean_height_m=meta.get('height_stats', {}).get('mean_m', 0.0),
                has_mesh=meta.get('products', {}).get('mesh_glb', False),
                has_pointcloud=meta.get('products', {}).get('pointcloud_ply', False),
                model_identity=meta.get('model', {}).get('identity')
            ))
    return sorted(
        results,
        key=lambda item: (item.model_identity == settings.MODEL_NAME, item.created_at),
        reverse=True,
    )

@router.get("/{scene_id}", response_model=SceneMetadataResponse, summary="Get Scene Metadata")
@router.get("/{scene_id}/metadata", response_model=SceneMetadataResponse, summary="Get Scene Metadata (Alias)")
async def get_scene_metadata(scene_id: str):
    """
    Returns full spatial metadata, elevation statistics, and available product URLs.
    """
    meta = scene_service.load_scene_metadata(scene_id)
    if not meta:
        raise SceneNotFoundError(scene_id)
    return SceneMetadataResponse(**meta)

@router.get("/{scene_id}/mesh.glb", summary="Stream an AGL or Absolute DSM 3D GLB Surface")
async def get_mesh_glb(
    scene_id: str,
    surface: str = Query("agl", pattern="^(agl|absolute_dsm)$"),
):
    """
    Streams the textured WebGL 3D surface in GLTF 2.0 Binary format.

    ``surface=agl`` is always the default. ``surface=absolute_dsm`` is only
    available when an aligned terrain/base DEM was supplied for the scene.
    """
    scene_dir = scene_service.get_scene_dir(scene_id)
    metadata = scene_service.load_scene_metadata(scene_id)
    if metadata is None:
        raise SceneNotFoundError(scene_id)
    recorded_surface = metadata.get("output_semantics", {}).get(
        "mesh_height_surface", "predicted_agl"
    )
    if surface in ("relative", "relative_surface"):
        candidates = [
            os.path.join(scene_dir, "mesh", "scene_relative.glb"),
            os.path.join(scene_dir, "mesh", "scene.glb"),
        ]
    elif surface == "absolute_dsm":
        candidates = [os.path.join(scene_dir, "mesh", "scene_absolute_dsm.glb")]
        if recorded_surface == "absolute_dsm":
            candidates.append(os.path.join(scene_dir, "mesh", "scene.glb"))
    else:
        candidates = [os.path.join(scene_dir, "mesh", "scene_agl.glb")]
        if recorded_surface != "absolute_dsm":
            candidates.append(os.path.join(scene_dir, "mesh", "scene.glb"))

    glb_path = next((path for path in candidates if os.path.exists(path)), "")

    # Fallback for legacy AGL scenes with a directly named GLB.
    if surface in ("agl", "relative") and not glb_path:
        candidates = [f for f in os.listdir(scene_dir) if f.endswith('.glb')] if os.path.exists(scene_dir) else []
        if candidates:
            glb_path = os.path.join(scene_dir, candidates[0])
            
    if not os.path.exists(glb_path):
        if surface == "absolute_dsm":
            raise AssetNotAvailableError("Absolute DSM mesh requires an aligned terrain DEM")
        if surface in ("relative", "relative_surface"):
            raise AssetNotAvailableError("Relative Surface mesh.glb")
        raise AssetNotAvailableError("AGL mesh.glb")
        
    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=f"{scene_id}_{surface}_mesh.glb",
    )

@router.get("/{scene_id}/pointcloud.ply", summary="Download Dense 3D Point Cloud")
async def get_pointcloud_ply(scene_id: str):
    """
    Downloads the colored dense 3D point cloud in standard binary PLY format.
    """
    scene_dir = scene_service.get_scene_dir(scene_id)
    ply_path = os.path.join(scene_dir, "pointcloud", "scene.ply")
    
    if not os.path.exists(ply_path):
        candidates = [f for f in os.listdir(scene_dir) if f.endswith('.ply')] if os.path.exists(scene_dir) else []
        if candidates:
            ply_path = os.path.join(scene_dir, candidates[0])

    if not os.path.exists(ply_path):
        raise AssetNotAvailableError("pointcloud.ply")
        
    return FileResponse(ply_path, media_type="application/octet-stream", filename=f"{scene_id}_pointcloud.ply")

@router.get("/{scene_id}/texture/rgb", summary="Get Optical RGB Texture")
async def get_texture_rgb(scene_id: str):
    scene_dir = scene_service.get_scene_dir(scene_id)
    path = os.path.join(scene_dir, "textures", "rgb.jpg")
    if not os.path.exists(path):
        path = os.path.join(scene_dir, "texture_rgb.jpg")
    if not os.path.exists(path):
        raise AssetNotAvailableError("texture_rgb.jpg")
    return FileResponse(path, media_type="image/jpeg")

@router.get("/{scene_id}/heightmap", summary="Get Height Heatmap Texture")
async def get_heightmap_texture(scene_id: str):
    scene_dir = scene_service.get_scene_dir(scene_id)
    path = os.path.join(scene_dir, "textures", "height_heatmap.png")
    if not os.path.exists(path):
        path = os.path.join(scene_dir, "texture_height_heatmap.jpg")
    if not os.path.exists(path):
        raise AssetNotAvailableError("height_heatmap.png")
    return FileResponse(path, media_type="image/png" if path.endswith('.png') else "image/jpeg")

@router.get("/{scene_id}/relative_surface", summary="Get Relative Surface Texture")
async def get_relative_surface_texture(scene_id: str):
    scene_dir = scene_service.get_scene_dir(scene_id)
    path = os.path.join(scene_dir, "textures", "relative_surface.png")
    if not os.path.exists(path):
        raise AssetNotAvailableError("relative_surface.png")
    return FileResponse(path, media_type="image/png")

@router.get("/{scene_id}/slope", summary="Get Slope Heatmap Texture")
async def get_slope_texture(scene_id: str):
    scene_dir = scene_service.get_scene_dir(scene_id)
    path = os.path.join(scene_dir, "textures", "slope.png")
    if not os.path.exists(path):
        path = os.path.join(scene_dir, "texture_slope_heatmap.jpg")
    if not os.path.exists(path):
        raise AssetNotAvailableError("slope.png")
    return FileResponse(path, media_type="image/png" if path.endswith('.png') else "image/jpeg")

@router.get("/{scene_id}/download/{asset_name}", summary="Direct Secure Artifact Download")
async def download_artifact(scene_id: str, asset_name: str):
    """
    Securely downloads scene artifacts (e.g. agl.tif, dsm.tif, mesh.glb, pointcloud.ply).
    """
    scene_dir = scene_service.get_scene_dir(scene_id)
    if not os.path.exists(scene_dir):
        raise SceneNotFoundError(scene_id)

    # Sanitize asset name to prevent path traversal
    safe_name = os.path.basename(asset_name)
    
    # Map common aliases to subdirectories
    file_map = {
        'agl.tif': os.path.join(scene_dir, "rasters", "predicted_agl.tif"),
        'predicted_agl.tif': os.path.join(scene_dir, "rasters", "predicted_agl.tif"),
        'dsm.tif': os.path.join(scene_dir, "rasters", "absolute_dsm.tif"),
        'absolute_dsm.tif': os.path.join(scene_dir, "rasters", "absolute_dsm.tif"),
        'aligned_terrain_dem.tif': os.path.join(scene_dir, "rasters", "aligned_terrain_dem.tif"),
        'terrain_dem.tif': os.path.join(scene_dir, "rasters", "aligned_terrain_dem.tif"),
        'dem.tif': os.path.join(scene_dir, "rasters", "aligned_terrain_dem.tif"),
        'predicted_agl.npy': os.path.join(scene_dir, "rasters", "predicted_agl.npy"),
        'absolute_dsm.npy': os.path.join(scene_dir, "rasters", "absolute_dsm.npy"),
        'base_dem.npy': os.path.join(scene_dir, "rasters", "base_dem.npy"),
        'relative_surface.npy': os.path.join(scene_dir, "rasters", "relative_surface.npy"),
        'relative_surface.png': os.path.join(scene_dir, "textures", "relative_surface.png"),
        'mesh.glb': os.path.join(scene_dir, "mesh", "scene.glb"),
        'scene.glb': os.path.join(scene_dir, "mesh", "scene.glb"),
        'scene_agl.glb': os.path.join(scene_dir, "mesh", "scene_agl.glb"),
        'scene_relative.glb': os.path.join(scene_dir, "mesh", "scene_relative.glb"),
        'scene_absolute_dsm.glb': os.path.join(scene_dir, "mesh", "scene_absolute_dsm.glb"),
        'mesh.obj': os.path.join(scene_dir, "mesh", "scene.obj"),
        'pointcloud.ply': os.path.join(scene_dir, "pointcloud", "scene.ply"),
        'scene.ply': os.path.join(scene_dir, "pointcloud", "scene.ply")
    }

    target_path = file_map.get(safe_name, os.path.join(scene_dir, safe_name))
    if not os.path.exists(target_path):
        raise AssetNotAvailableError(safe_name)

    return FileResponse(target_path, filename=f"{scene_id}_{safe_name}")
