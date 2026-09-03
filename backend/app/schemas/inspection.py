"""
DepthWizard (SIH26175) — Inspection & Measurement Schemas
Player 4: Backend & Systems Integration Lead
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class InspectRequest(BaseModel):
    u: int = Field(..., ge=0, description="Horizontal pixel coordinate (column 0..W-1)")
    v: int = Field(..., ge=0, description="Vertical pixel coordinate (row 0..H-1)")

class InspectResponse(BaseModel):
    scene_id: str
    pixel: List[int] = Field(..., description="[u, v] queried pixel index")
    predicted_agl_m: float = Field(..., description="Authoritative full-resolution AGL height in metres")
    absolute_elevation_m: Optional[float] = Field(None, description="Absolute DSM elevation when an exactly aligned terrain DEM is available")
    semantic_class: Optional[str] = Field(None, description="Only populated when a real semantic model output exists")
    slope_deg: Optional[float] = Field(None, description="Physical terrain slope angle in degrees")
    world_coordinates_m: Optional[Dict[str, float]] = Field(None, description="Local metric coordinates when horizontal GSD is known")
    is_georeferenced: bool
    warnings: List[str] = Field(default_factory=list)

class MeasureRequest(BaseModel):
    point_a: List[float] = Field(..., description="[u1, v1] or [x1, y1, z1] start point")
    point_b: List[float] = Field(..., description="[u2, v2] or [x2, y2, z2] end point")
    is_pixel_coords: bool = Field(True, description="True if coordinates are in [u, v] pixels, False if 3D metric world [x, y, z]")

class MeasureResponse(BaseModel):
    scene_id: str
    horizontal_distance_m: float = Field(..., description="Horizontal 2D ground baseline distance in metres")
    direct_3d_distance_m: float = Field(..., description="Direct 3D Euclidean distance in metres")
    elevation_delta_m: float = Field(..., description="Vertical elevation difference (h2 - h1) in metres")
    slope_angle_deg: float = Field(..., description="Direct line-of-sight slope angle in degrees")
    start_elevation_m: float
    end_elevation_m: float
    warnings: List[str] = Field(default_factory=list)
