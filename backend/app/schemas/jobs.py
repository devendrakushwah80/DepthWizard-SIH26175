"""
DepthWizard (SIH26175) — Job Schemas
Player 4: Backend & Systems Integration Lead
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class JobStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING_INFERENCE = "running_inference"
    PROCESSING_GEOSPATIAL = "processing_geospatial"
    GENERATING_3D = "generating_3d"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobCreateResponse(BaseModel):
    job_id: str = Field(..., description="Unique UUID for tracking the asynchronous processing job")
    status: JobStatus = Field(JobStatus.QUEUED, description="Initial job status")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    message: str = Field("Job queued successfully", description="Status message")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique job UUID")
    scene_id: Optional[str] = Field(None, description="Generated scene identifier")
    status: JobStatus = Field(..., description="Current processing state")
    progress_pct: int = Field(..., ge=0, le=100, description="Stage-based progress percentage")
    current_stage: str = Field(..., description="Human-readable stage description")
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_s: Optional[float] = None
    model_identity: str = "M3-FINAL"
    warnings: List[str] = Field(default_factory=list)
    processing_timings: Optional[Dict[str, float]] = None
    peak_vram_mb: Optional[float] = None
    artifact_sizes_bytes: Optional[Dict[str, int]] = None
