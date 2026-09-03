# DepthWizard Reproducibility Guide

**Problem statement:** SIH26175 — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space

## Supported environment

- Windows 10/11 or Linux
- Python 3.10+
- Node.js 20+
- NVIDIA CUDA GPU recommended; CPU fallback is supported

Python packages are pinned in `requirements.txt` and `requirements-dev.txt`.
Frontend packages are locked by `frontend/package-lock.json`.

## Clean setup

```powershell
git clone https://github.com/devendrakushwah80/DepthWizard-SIH26175.git
cd DepthWizard-SIH26175
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
cd frontend
npm ci
cd ..
```

The pretrained DAV2 prior may be downloaded from Hugging Face on first use. Raw
GAMUS, cached DAV2 arrays, SRTM files, and generated outputs are intentionally not
stored in Git.

## Launch

```powershell
.\scripts\start_depthwizard.ps1
```

- Frontend: `http://localhost:3000/`
- API documentation: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/health`

Stop the launcher with `Ctrl+C`. Individual server processes can also be stopped
from the terminal in which they were started.

## Checkpoint verification

The release checkpoint is `models/m2_final/M2_FINAL.pth`.

```powershell
(Get-FileHash -Algorithm SHA256 models\m2_final\M2_FINAL.pth).Hash.ToLower()
```

Expected SHA256:

```text
6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f
```

The backend verifies this digest and does not silently fall back to another
trained checkpoint.

## Automated verification

```powershell
python -m pytest -q backend/tests tests
cd frontend
npm test
npm run build
```

The backend suite uses deterministic temporary inputs and a deterministic model
stub for API pipeline tests. Model-health verification separately checks the real
sealed checkpoint identity. This distinction avoids presenting a hermetic API
test as a full learned-model accuracy evaluation.

## Data and output reproduction

Deterministic GAMUS split manifests are stored in `data/gamus/splits/`. Place
locally acquired raw data under `data/gamus/dataset/`; that directory is ignored.
Training and evaluation scripts write into `outputs/`, which is also ignored.

Compact historical results used by the public documentation are stored under
`docs/evidence/`. Full-resolution arrays and visual artifacts are intentionally
kept out of Git.

## Scientific invariants

- The neural network predicts AGL/nDSM, not absolute terrain elevation.
- Absolute DSM is available only as `aligned DEM + predicted AGL`.
- Raster row `v=0` maps to positive mesh Z and UV `V=1`.
- Generated scientific geometry remains at 1x physical vertical scale.
- Viewer vertical exaggeration is display-only.
