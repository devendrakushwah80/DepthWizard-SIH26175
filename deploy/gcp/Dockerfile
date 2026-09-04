# DepthWizard (SIH26175) — Google Cloud Run Production Dockerfile
# CPU-Optimized Inference Service for ISRO / Department of Space

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEPTHWIZARD_DEVICE=cpu \
    PORT=8080

# Install minimal OS dependencies for GDAL/Rasterio and OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Depth Anything V2 Small model weights into container image cache
# Eliminates external network dependency during Cloud Run container cold-start
RUN python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation; AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')"

# Copy core application code, configurations, seed demo scene, and production weights
COPY backend ./backend
COPY src ./src
COPY configs ./configs
COPY seed_scenes ./seed_scenes
COPY models/m3_final ./models/m3_final
COPY models/m2_final ./models/m2_final

EXPOSE 8080

# Start FastAPI server respecting Cloud Run dynamic $PORT
CMD exec sh -c "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080}"
