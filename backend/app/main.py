"""
DepthWizard (SIH26175) — FastAPI Application Server
Player 4: Backend & Systems Integration Lead
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.core.errors import DepthWizardError, depthwizard_exception_handler, generic_exception_handler
from backend.app.services.model_service import model_service
from backend.app.api import routes_health, routes_jobs, routes_scenes, routes_inspect

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server Startup: Initialize resident PyTorch models once into GPU memory
    print("=" * 80)
    print(f"STARTING {settings.PROJECT_NAME} v{settings.VERSION}")
    print("=" * 80)
    model_service.initialize()
    print("=" * 80)
    print("SYSTEM READY FOR API REQUESTS")
    print("=" * 80)
    yield
    # Server Shutdown
    print("[Server] Shutting down DepthWizard API...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
# DepthWizard — Single-View Satellite Height Estimation & 3D Flythrough API
**Problem Statement ID:** SIH26175 (ISRO / Department of Space)

### Core Capabilities:
* **AI Height Estimation:** Sealed M2-FINAL with a frozen Depth Anything V2 Small relative-depth prior.
* **Primary Output:** Predicted metric AGL / nDSM (height above local ground).
* **Geospatial Processing:** GeoTIFF CRS/Affine preservation and strictly aligned Base DEM + AGL $\to$ Absolute DSM computation.
* **3D Surface Reconstruction:** Textured physical-scale `.glb` AGL/DSM surface meshes ($384\times 384$), dense `.ply` point clouds, and multi-mode heatmaps.
* **Interactive Tooling:** Full-resolution pixel raycast height inspection and 3D spatial ruler measurements.
""",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# Configure CORS for Frontend Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Custom Exception Handlers
app.add_exception_handler(DepthWizardError, depthwizard_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register API Routers
app.include_router(routes_health.router)
app.include_router(routes_jobs.router)
app.include_router(routes_scenes.router)
app.include_router(routes_inspect.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
