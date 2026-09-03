# DepthWizard (SIH26175) — SIH Requirement Traceability
**Problem Statement ID:** SIH26175  
**Title:** DepthWizard — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space  

---

## 1. 50% DSM / Height Estimation Accuracy & Scientific Validation

### 1.1 Model Architecture & Foundation Prior
* **Requirement:** Exploit foundational depth features for monocular height estimation.
* **Implementation:** `src/models/dav2_wrapper.py` + `src/models/rdah_net.py` (Sealed Model: `M2-FINAL`, architecture `RDAHNetCore`, 5,372,047 trainable parameters, non-negative `softplus` output parameterization, combined with frozen `depth-anything/Depth-Anything-V2-Small-hf` prior).
* **Identity & Integrity:**
  * Checkpoint: `models/m2_final/M2_FINAL.pth`
  * Checkpoint SHA256: `6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f`
  * Selection: Trial 10, Seed 1337, Epoch 10 (selected strictly on validation dev set without NYC exposure).
* **Status:** `PASS`

### 1.2 Quantitative Regression Metrics
* **Requirement:** Scientifically rigorous error evaluation across independent validation and held-out zero-shot test sets.
* **Implementation:** `scripts/player6_master_evaluation.py` and evaluation pipeline.
* **Evidence:**
  * **Final-Dev Validation (200 tiles, synthetic DDSM/WHU-Mix & GAMUS):**
    * $\text{MAE} = 2.5175966461\text{ m}$
    * $\text{RMSE} = 4.7376244827\text{ m}$
    * $R^2 = 0.6309778287$
    * $\text{Pearson } r = 0.8062702933$
    * $\text{Bias} = -0.9176289402\text{ m}$
  * **Held-Out Zero-Shot NYC Benchmark (496 tiles, 478,021,075 pixels):**
    *(Note: NYC was strictly held out and never used for training, hyperparameter search, or calibration)*
    * Overall All-Pixel: $\text{MAE} = 4.844\text{ m}$, $\text{RMSE} = 8.144\text{ m}$, Pearson $r = 0.272$, $R^2 = -0.222$, $\text{Bias} = -3.141\text{ m}$, $\text{Median AE} = 1.564\text{ m}$, $\text{NMAD} = 2.322\text{ m}$
    * Error distribution: Within $1\text{m} = 45.54\%$, Within $2\text{m} = 52.79\%$, Within $5\text{m} = 66.61\%$, Within $10\text{m} = 81.01\%$
    * Building-specific subset ($66,797,828\text{ pixels}$): $\text{MAE} = 4.388\text{ m}$, $\text{RMSE} = 6.415\text{ m}$, Pearson $r = 0.335$, $\text{Bias} = -0.423\text{ m}$, $\text{Median AE} = 3.222\text{ m}$, Within $5\text{m} = 67.67\%$, Within $10\text{m} = 91.86\%$
    * Height Stratification:
      * $0\text{--}2\text{ m}$ ($50.96\%$ of pixels): $\text{MAE} = 0.985\text{ m}$, $\text{RMSE} = 2.672\text{ m}$, $\text{Bias} = +0.771\text{ m}$, Within $1\text{m} = 82.49\%$
      * $2\text{--}10\text{ m}$ ($25.60\%$ of pixels): $\text{MAE} = 4.425\text{ m}$, $\text{RMSE} = 5.294\text{ m}$, $\text{Bias} = -1.745\text{ m}$, Within $5\text{m} = 61.22\%$
      * $10\text{--}20\text{ m}$ ($16.97\%$ of pixels): $\text{MAE} = 11.246\text{ m}$, $\text{RMSE} = 12.425\text{ m}$, $\text{Bias} = -10.923\text{ m}$
      * $>20\text{ m}$ ($6.46\%$ of pixels): $\text{MAE} = 24.364\text{ m}$, $\text{RMSE} = 27.094\text{ m}$, $\text{Bias} = -24.237\text{ m}$ (upper-tail regression compression on extreme skyscrapers)
* **Status:** `PASS`

### 1.3 Landscape & Domain Robustness
* **Requirement:** Performance stability across diverse terrain morphology (urban, sparse, hilly, forested).
* **Implementation:** `data/evaluation/domain_*.txt`.
* **Evidence:**
  * **Urban & Suburban:** `PASS` (Over 696 tiles evaluated across Philadelphia, Washington DC, and New York City).
  * **Ground & Low-Rise ($<2\text{ m}$):** `PASS` ($\text{MAE} < 1.0\text{ m}$ across $>243\text{M}$ pixels).
  * **Hilly / Mountainous Topography:** `NOT VALIDATED` (No mountainous LiDAR ground truth in benchmark dataset; explicitly documented without fabrication).
  * **Dense Wild Forest:** `NOT VALIDATED` (Urban tree canopies validated; continuous wild canopies not benchmarked).
* **Status:** `PARTIAL (Fully Documented per Scientific Protocol)`

---

## 2. 50% Visualization, Rendering & Software Stability

### 2.1 3D Reconstruction & Watertight Mesh
* **Requirement:** Watertight 3D terrain surface reconstruction with RGB texture mapping.
* **Implementation:** `src/geometry/raster_to_mesh.py` + `raster_to_pointcloud.py`.
* **Evidence:** $384\times 384$ binary GLB terrain model ($10.06\text{ MB}$, $296\text{k}$ triangles) with ground skirts; mesh resampling deviation $< 1.9\text{ cm}$. Supports Relative Surface, Predicted AGL, and Absolute DSM modes.
* **Status:** `PASS`

### 2.2 Interactive 3D Flythrough
* **Requirement:** Smooth 60 FPS first-person camera flythrough.
* **Implementation:** `frontend/src/components/SceneWorkspace/ThreeDCanvas.tsx` (Three.js `PointerLockControls` and `OrbitControls`).
* **Evidence:** Real-time 60 FPS flight mode (WASD + mouse look + Q/E altitude) and deterministic Image-Aligned / Top / Oblique / Eye camera presets.
* **Status:** `PASS`

### 2.3 Authoritative Height & Spatial Ruler
* **Requirement:** Interactive measurement tools for analytical geospatial workflows.
* **Implementation:** `backend/app/services/scene_service.py` (`/inspect` and `/measure`).
* **Evidence:** Full raster raycasting returning metric AGL, absolute elevation (when DEM is available), slope, and 3D Euclidean distances. Uses authoritative physical arrays, unaffected by display vertical exaggeration.
* **Status:** `PASS`

### 2.4 Standalone Deployment & Backend Reliability
* **Requirement:** Production-ready API and standalone one-command execution.
* **Implementation:** `scripts/start_depthwizard.ps1` / `scripts/start_depthwizard.py` + `backend/app/main.py`.
* **Evidence:** Automated startup preflight, sealed checkpoint SHA-256 verification, and hermetic regression test suite.
* **Status:** `PASS`

