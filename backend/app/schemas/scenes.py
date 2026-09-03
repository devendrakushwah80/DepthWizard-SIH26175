"""
DepthWizard (SIH26175) — Scene Schemas
Player 4: Backend & Systems Integration Lead
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SceneInputInfo(BaseModel):
    format: str = Field(..., description="PNG, JPEG, GeoTIFF, or H5")
    width: int
    height: int
    channels: int = 3
    is_georeferenced: bool = False

class SpatialInfo(BaseModel):
    is_georeferenced: bool = False
    crs: Optional[str] = None
    gsd_m: Optional[float] = None
    gsd_source: str = "unavailable"
    horizontal_units: str = "pixels"
    physical_width_m: Optional[float] = None
    physical_height_m: Optional[float] = None
    bounds: Optional[Dict[str, float]] = None
    transform: Optional[List[float]] = None
    resolution: Optional[List[float]] = None

class HeightStatistics(BaseModel):
    type: str = "AGL"
    unit: str = "metres"
    min_m: float
    max_m: float
    mean_m: float
    median_m: float
    std_m: float

class SceneProducts(BaseModel):
    predicted_agl_npy: bool = True
    predicted_agl_tif: bool = False
    absolute_dsm_tif: bool = False
    mesh_glb: bool = True
    mesh_agl_glb: bool = False
    mesh_absolute_dsm_glb: bool = False
    mesh_obj: bool = True
    pointcloud_ply: bool = True
    texture_rgb: bool = True
    texture_height_heatmap: bool = True
    texture_semantic: bool = False
    texture_slope_heatmap: bool = True

class SceneMetadataResponse(BaseModel):
    scene_id: str
    status: str = "completed"
    created_at: str
    input_info: SceneInputInfo
    spatial_info: SpatialInfo
    height_stats: HeightStatistics
    slope_mean_deg: Optional[float] = None
    slope_max_deg: Optional[float] = None
    products: SceneProducts
    artifacts: Dict[str, Optional[str]]
    model: Optional[Dict[str, Any]] = None
    output_semantics: Optional[Dict[str, Any]] = None
    processing_timings: Dict[str, float] = Field(default_factory=dict)
    peak_vram_mb: float = 0.0
    artifact_sizes_bytes: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    mesh_info: Dict[str, Any] = Field(default_factory=dict)
    pointcloud_info: Dict[str, Any] = Field(default_factory=dict)

class SceneListItem(BaseModel):
    scene_id: str
    created_at: str
    input_format: str
    is_georeferenced: bool
    max_height_m: float
    mean_height_m: float
    has_mesh: bool
    has_pointcloud: bool
    model_identity: Optional[str] = None
