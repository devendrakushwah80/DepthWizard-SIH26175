# DepthWizard (SIH26175) — Known Limitations Register
**Problem Statement ID:** SIH26175  
**Title:** DepthWizard — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space  

---

## 1. High-Rise Upper-Tail Compression

* **Description:** Extreme vertical structures ($> 20\text{m}$ and $> 50\text{m}$) are underestimated (mean predicted height $\approx 5.34\text{m}$ vs ground truth $\approx 66.97\text{m}$ for $> 50\text{m}$ structures).
* **Root Cause:** Ground-truth training distribution is overwhelmingly ground/low-rise ($> 76\%$ pixels $< 2\text{m}$), leading to strong regression shrinkage toward the median.
* **Current Mitigation:** Full transparency in UI model card, elevation colorbars, and statistical metrics; Smooth L1 loss limits extreme gradient explosion.
* **Future Work:** Asymmetric log-scaled loss, focal height reweighting, and high-rise fine-tuning subsets.

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
* **Current Mitigation:** Application prompts user for optional GSD. If unsupplied, physical horizontal distances and metric slope are marked approximate, never fabricating fake $0.5\text{ m/px}$ or fake geographic coordinates.

---

## 5. Unrepresented Mountainous & Wild Forest Domains

* **Description:** The GAMUS training and validation dataset covers urban/suburban coastal cities (Philadelphia, Washington DC, New York City).
* **Current Mitigation:** Hilly and Dense Wild Forest categories are explicitly designated `NOT VALIDATED` in compliance with scientific integrity rules.
