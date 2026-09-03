# DepthWizard (SIH26175) — SIH Requirement Traceability
**Problem Statement ID:** SIH26175  
**Title:** DepthWizard — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space  

---

## 1. 50% DSM Estimation Accuracy & Validation

### 1.1 Model Architecture & Foundation Prior
* **Requirement:** Exploit foundational depth features for monocular height estimation.
* **Implementation:** `src/models/dav2_wrapper.py` + `src/models/rdah_net.py` (M1: RDAH-Net + Frozen Depth Anything V2 Small).
* **Evidence:** Stage A1 comparative training curves and zero-shot baseline reports.
* **Result:** M1 beats RGB-only on $77.02\%$ of unseen NYC tiles ($p = 3.89 \times 10^{-36}$).
* **Status:** `PASS`

### 1.2 Quantitative Regression Metrics
* **Requirement:** Scientifically rigorous error evaluation across independent test sets.
* **Implementation:** `scripts/player6_master_evaluation.py`.
* **Evidence:**
  * **PHL/DC Validation (200 tiles):** MAE $= 3.308\text{ m}$, RMSE $= 6.275\text{ m}$, Pearson $r = 0.680$, $R^2 = 0.442$, NMAD $= 1.842\text{ m}$.
  * **Held-Out NYC Benchmark (496 tiles):** MAE $= 4.912\text{ m}$, RMSE $= 8.356\text{ m}$, Pearson $r = 0.249$, Building MAE $= 5.023\text{ m}$, Building $\pm 5\text{m} = 64.47\%$.
* **Status:** `PASS`

### 1.3 Landscape & Domain Robustness
* **Requirement:** Performance stability across diverse terrain morphology (urban, sparse, hilly, forested).
* **Implementation:** `data/evaluation/domain_*.txt`.
* **Evidence:**
  * **Urban:** `PASS` (500 NYC/PHL/DC tiles evaluated).
  * **Sparse / Suburban:** `PASS` (50 fringe tiles evaluated, MAE $= 3.12\text{ m}$).
  * **Hilly / Mountainous:** `NOT VALIDATED` (No mountainous LiDAR in GAMUS; explicitly preserved without fabrication).
  * **Dense Forest:** `NOT VALIDATED` (Urban vegetation validated; dense continuous wild canopies unavailable).
* **Status:** `PARTIAL (Documented)`

---

## 2. 50% Visualization, Rendering & Software Stability

### 2.1 3D Reconstruction & Watertight Mesh
* **Requirement:** Watertight 3D terrain surface reconstruction with RGB texture mapping.
* **Implementation:** `src/geometry/raster_to_mesh.py` + `raster_to_pointcloud.py`.
* **Evidence:** $384\times 384$ binary GLB terrain model ($10.06\text{ MB}$, $296\text{k}$ triangles) with ground skirts; mesh resampling deviation $< 1.9\text{ cm}$.
* **Status:** `PASS`

### 2.2 Interactive 3D Flythrough
* **Requirement:** Smooth 60 FPS first-person camera flythrough.
* **Implementation:** `frontend/src/components/SceneWorkspace/ThreeDCanvas.tsx` (Three.js `PointerLockControls`).
* **Evidence:** Real-time 60 FPS drone flight mode (WASD + mouse look + Q/E altitude).
* **Status:** `PASS`

### 2.3 Authoritative Height & Spatial Ruler
* **Requirement:** Interactive measurement tools for analytical geospatial workflows.
* **Implementation:** `backend/app/services/scene_service.py` (`/inspect` and `/measure`).
* **Evidence:** Full $1024\times 1024$ raster raycasting returning metric AGL, absolute elevation, slope, and 3D Euclidean distances.
* **Status:** `PASS`

### 2.4 Standalone Deployment & Backend Reliability
* **Requirement:** Production-ready API and standalone one-command execution.
* **Implementation:** `scripts/start_depthwizard.ps1` + `backend/app/main.py`.
* **Evidence:** 16/16 functional test matrix passed; 10/10 sequential stress runs passed with zero memory leak ($123.3\text{ MB}$ VRAM).
* **Status:** `PASS`
