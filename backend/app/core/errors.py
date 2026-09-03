"""
DepthWizard (SIH26175) — Error Handling & Custom Exceptions
Player 4: Backend & Systems Integration Lead
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

class DepthWizardError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class InvalidFileTypeError(DepthWizardError):
    def __init__(self, message: str = "Uploaded file format is unsupported"):
        super().__init__(code="INVALID_FILE_TYPE", message=message, status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

class UnreadableImageError(DepthWizardError):
    def __init__(self, message: str = "Image file is corrupt or unreadable"):
        super().__init__(code="UNREADABLE_IMAGE", message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class InvalidGeoTIFFError(DepthWizardError):
    def __init__(self, message: str = "GeoTIFF metadata is corrupt, invalid, or missing required bands"):
        super().__init__(code="INVALID_GEOTIFF", message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class ModelInferenceError(DepthWizardError):
    def __init__(self, message: str = "Height estimation model inference failed"):
        super().__init__(code="MODEL_INFERENCE_FAILED", message=message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SceneNotFoundError(DepthWizardError):
    def __init__(self, scene_id: str):
        super().__init__(code="SCENE_NOT_FOUND", message=f"Scene '{scene_id}' not found", status_code=status.HTTP_404_NOT_FOUND)

class JobNotFoundError(DepthWizardError):
    def __init__(self, job_id: str):
        super().__init__(code="JOB_NOT_FOUND", message=f"Job '{job_id}' not found", status_code=status.HTTP_404_NOT_FOUND)

class AssetNotAvailableError(DepthWizardError):
    def __init__(self, asset_name: str):
        super().__init__(code="ASSET_NOT_AVAILABLE", message=f"Requested asset '{asset_name}' is not generated for this scene", status_code=status.HTTP_404_NOT_FOUND)

async def depthwizard_exception_handler(request: Request, exc: DepthWizardError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )

async def generic_exception_handler(request: Request, exc: Exception):
    # Log internal error on server without leaking raw trace to user
    print(f"[INTERNAL SERVER ERROR] {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while processing the request.",
                "details": {}
            }
        }
    )
