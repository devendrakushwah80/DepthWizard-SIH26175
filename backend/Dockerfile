# DepthWizard (SIH26175) — Backend Dockerfile
# CPU Fallback / Containerized Inference Service

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for GDAL/Rasterio and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained weights
COPY backend ./backend
COPY src ./src
COPY configs ./configs
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
