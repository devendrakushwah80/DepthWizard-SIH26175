# DepthWizard — Single-View Height Estimation and 3D Flythrough

DepthWizard is an end-to-end research application for generating an AI-estimated
height surface from a single optical remote-sensing image and exploring that
surface as an interactive textured 3D mesh. It was developed for Smart India
Hackathon problem statement **SIH26175** from ISRO / Department of Space.

## Scientific output semantics

- PNG/JPG input: produces a non-georeferenced predicted AGL/rDSM product. If no
  ground sample distance (GSD) is supplied, horizontal metric distance and slope
  are reported as unavailable rather than inferred.
- Georeferenced GeoTIFF without a DEM: preserves the optical CRS/grid and exports
  predicted AGL/nDSM. This is not absolute terrain elevation.
- Georeferenced GeoTIFF with an exactly aligned terrain DEM: exports
  `absolute DSM = aligned DEM + predicted AGL`.
- All generated raster and mesh elevations remain at physical **1x** scale.
  Viewer options such as 2x, 5x, and 10x are display-only and never change
  inspection, ruler, AGL, DSM, or exported scientific values.

The monocular model is not a replacement for LiDAR, stereo photogrammetry, or a
survey-grade DEM. See [Known Limitations](docs/KNOWN_LIMITATIONS.md).

## Included components

- PyTorch M3-FINAL multi-domain regression model with GSD FiLM conditioning and frozen Depth Anything V2 Small prior
- Sealed M2-FINAL baseline checkpoint preserved for comparative traceability
- FastAPI upload, inference, geospatial validation, inspection, measurement, and scene-asset APIs
- React, Three.js, and Vite frontend with 2D comparison and interactive 3D views
- GLB/OBJ mesh, PLY point-cloud, AGL GeoTIFF, and optional absolute DSM GeoTIFF export
- Deterministic data splits, model manifest, tests, and lightweight evaluation evidence

## Requirements

- Python 3.10 or newer
- Node.js 20 or newer with npm
- NVIDIA CUDA GPU recommended; CPU mode is supported but slower
- Internet access on the first run to obtain the configured frozen
  `depth-anything/Depth-Anything-V2-Small-hf` prior if it is not cached

The production M3-FINAL checkpoint is located at `models/m3_final/M3_FINAL.pth`:
```text
db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330
```

The sealed M2-FINAL baseline checkpoint is preserved at `models/m2_final/M2_FINAL.pth`:
```text
6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f
```

## Setup

### Windows PowerShell

```powershell
git clone https://github.com/devendrakushwah80/DepthWizard-SIH26175.git
cd DepthWizard-SIH26175
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
cd ..
.\scripts\start_depthwizard.ps1
```

### Linux/macOS

```bash
git clone https://github.com/devendrakushwah80/DepthWizard-SIH26175.git
cd DepthWizard-SIH26175
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd frontend && npm ci && cd ..
python scripts/start_depthwizard.py
```

The frontend opens at `http://localhost:3000`; API documentation is available at
`http://localhost:8000/docs`.

To configure CPU mode or custom storage/model locations, copy `.env.example` and
set the corresponding `DEPTHWIZARD_*` environment variables before launching.

### Docker support

Container configuration included; build not validated in this release.

## Validation

```powershell
# Backend, geospatial, geometry, and pipeline tests
python -m pytest -q backend/tests tests

# Frontend unit/viewer tests and production build
cd frontend
npm test
npm run build
```

Verify the production checkpoint on Windows:

```powershell
(Get-FileHash -Algorithm SHA256 models\m2_final\M2_FINAL.pth).Hash.ToLower()
```

## Repository data policy

Raw GAMUS HDF5 tiles, DAV2 caches, SRTM data, virtual environments, generated
scenes, training checkpoints, and full evaluation products are intentionally not
tracked. Lightweight deterministic splits are retained in `data/gamus/splits/`,
while compact public results are retained in `docs/evidence/`.

## Documentation

- [Model card](docs/MODEL_CARD.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Requirements traceability](docs/SIH_REQUIREMENT_TRACEABILITY.md)
- [GAMUS audit](docs/GAMUS_DATA_AUDIT.md)
- [Geospatial and DEM calibration](docs/GEOSPATIAL_SRTM_CALIBRATION_NOTE.md)
- [Public evidence index](docs/evidence/README.md)

## License

No open-source license has been declared in this repository. Obtain permission
from the repository owner before reuse or redistribution, and separately comply
with the terms of all datasets, pretrained models, and third-party dependencies.
