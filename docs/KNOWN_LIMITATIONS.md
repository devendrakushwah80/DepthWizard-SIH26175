# DepthWizard (SIH26175) — Known Limitations Register
**Problem Statement ID:** SIH26175  
**Title:** DepthWizard — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space  

---

## 1. High-Rise Upper-Tail Compression (Substantially Reduced; Extreme $\ge$50m Underestimation Remains)

* **Description:** Extreme vertical structures ($> 20\text{m}$ and $> 50\text{m}$) historically suffered from regression shrinkage in M2 due to natural class imbalance ($< 2\text{m}$ pixels account for $> 56\%$ of urban scenes).
* **M3-FINAL Mitigation:** Through height-aware tile sampling (oversampling high-rise regions 4x) and capped height-weighted composite loss ($1.0 + \min(0.35, y/50)$ with tall bias penalty), high-rise compression was substantially reduced:
  - 20–50m high-rise structure MAE dropped by **59.9%** on DEV (19.18m down to 7.69m) and **63.8%** on Holdout (17.85m down to 6.47m).
  - Skyscraper ($\ge 50\text{m}$) MAE was reduced from 53.33m down to 33.36m on Holdout, and from 35.83m down to 29.52m on NYC.
* **Remaining Limitation:** Extreme skyscrapers ($\ge 50\text{m}$ and megatall $> 100\text{m}$) remain heavily under-represented in nadir training data and exhibit residual negative shrinkage bias.

---

## 2. Monocular Single-View Ambiguity & Occlusion

* **Description:** Top-down single-view imagery lacks stereoscopic parallax. Narrow building alleys, shadowed courtyards, and vertical facades under direct nadir angles exhibit geometric ambiguity.
* **Current Mitigation:** Foundation depth features from Depth Anything V2 Small provide relative boundary cues; Hanning sliding-window blending prevents tile boundary seams.
* **Future Work:** Multi-angle sensor fusion (e.g. Cartosat-2 stereo / tri-stereo integration).

---

## 3. DEM Dependency for Absolute Elevation (AMSL)

* **Description:** The AI predicts height Above Ground Level (AGL / nDSM). It does not know regional Mean Sea Level (AMSL) elevation.
* **Current Mitigation:** Strict equation $\text{Absolute DSM} = \text{Base DEM} + \text{Predicted AGL}$. If no DEM is provided, absolute elevation is explicitly marked *"Unavailable — no base DEM"*.
* **Future Work:** Automatic global DEM tiling service (e.g. Cop-DEM / SRTM 30m automated API fetch).

---

## 4. Unknown GSD for Non-Georeferenced Imagery

* **Description:** Arbitrary PNG/JPG files lack spatial resolution metadata.
* **Current Mitigation:** Application prompts user for optional GSD. If unsupplied, physical horizontal distances and metric slope are marked approximate, never fabricating fake $0.5\text{ m/px}$ or fake geographic coordinates. M3-FINAL dynamically adapts via its GSD FiLM vector $[GSD_m, GSD_{known}]$.

---

## 5. Natural Mountainous & Wild Forest Domains (Ground-Truth Audit & Canopy Validation)

* **Prior Benchmark Audit & Root Cause:** An exhaustive scientific audit of the 10 NAIP / 3DEP Natural Benchmark AOIs revealed that the USGS `3DEPElevation/ImageServer` distributes exclusively bare-earth Digital Terrain Models (DTM/DEM). Top-surface DSM rasters were missing, and ground-truth AGL was fabricated as all zeros ($0.0\text{m}$) across all 2.62M pixels. When M3 correctly predicted tree heights (15–28m) over dense Appalachian or Pacific Northwest forests, the evaluator measured distance to $0.0\text{m}$, creating a false "15.8m error" while M2 (which heavily compressed heights near zero) appeared artificially closer to flat ground.
* **Validated LiDAR Ground-Truth Benchmark:** When evaluated against a scientifically curated, 100% unseen natural benchmark with true airborne LiDAR $n\text{DSM} = \text{DSM} - \text{DTM}$ across 90 scenes (23.6M pixels):
  - **Overall Natural MAE:** M2 = **2.6761m**, M3 = **2.8899m** (M3 is broadly comparable to M2 on the combined natural benchmark, while remaining slightly worse in MAE and exhibiting significant canopy-height underestimation).
  - **Pixels within 2m:** M3 = **68.31%** (superior to M2's 67.92%).
  - **Forest Canopy ($\ge$ 4m):** M2 = 5.413m, M3 = 5.762m (delta 0.35m, M3 exhibits strong negative bias of -5.47m).
  - **Mixed Vegetation (2–15m):** M2 = 2.534m, M3 = 2.649m (delta 0.11m).
  - **Bare / Sloped Terrain (< 1m):** M2 = 0.081m, M3 = 0.259m.
* **Recommendation:** For mountainous bare terrain without vegetation, AGL is legitimately near zero; macroscopic geomorphic relief belongs to the base DEM ($\text{DSM} = \text{DEM} + \text{AGL}$). For wild forests, M3 preserves useful canopy structure cues but currently underestimates absolute forest canopy height; validated forest MAE is 5.76m with a negative bias of 5.47m.

---

## 6. Monocular Relative Surface vs Metric Height

* **Description:** The Relative Surface ($r\text{DSM}$) extracted from the frozen Depth Anything V2 prior provides scale-agnostic monocular relief cues. It has no physical metric scale, datum, or vertical calibration.
* **Current Mitigation:** DepthWizard strictly decouples the Relative Surface from metric Predicted AGL ($n\text{DSM}$) and Absolute DSM. Relative Surface visualizations are normalized to $[0, 1]$ and never displayed with metric metre units or elevation claims.
