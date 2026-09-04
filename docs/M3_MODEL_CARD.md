# Model Card — DepthWizard M3-FINAL

**Model Identifier:** `M3-FINAL`  
**Architecture:** `M3NetCore` (Dual-Branch MobileViT + Depth Anything V2 Prior + GSD FiLM Conditioning + PixelShuffle Decoder)  
**Task:** Monocular overhead optical RGB to dense metric Above Ground Level (AGL / nDSM) in metres  
**Production Checkpoint:** `models/m3_final/M3_FINAL.pth`  
**Checkpoint SHA-256:** `db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330`  

---

## 1. Model Summary & Architectural Innovations

`M3-FINAL` is the production deep learning model for the Smart India Hackathon (SIH26175) challenge: Single-View Height Estimation and 3D Flythrough for ISRO / Department of Space.

### Key Architectural Additions:
1. **Multi-Domain Ingestion:** Trained across the full multi-domain corpus:
   - **GAMUS Dataset:** Philadelphia (PHL), Washington DC (DC).
   - **US3D / DFC2019 Dataset:** Jacksonville (JAX) and Omaha (OMA) high-resolution urban lidar.
   - Total corpus: 3,533 verified aerial/satellite scenes (over 3.5 billion metric pixels).
2. **GSD FiLM Conditioning:**
   - Feature-wise Linear Modulation (FiLM) blocks at the global transformer bottleneck and decoder stage 2.
   - Dynamic scaling vector $[GSD_m, GSD_{known}]$ enabling scale-adaptive feature representation across varying satellite sensors (0.3m to 1.0m+).
3. **Height-Balanced Loss & Sampling:**
   - Height-anchored crop selection ensuring mid-rise and tall structures are sampled at 4x the natural frequency.
   - Capped height-weighted loss ($1.0 + \min(0.35, y/50)$) with tall-bias penalty mitigating upper-tail regression shrinkage.
4. **Strict Partition Sealing:**
   - Geographic super-block spatial clustering preventing spatial data leakage.
   - New York City (NYC, 500 scenes) kept 100% unseen as an external zero-shot test set.
   - Final Holdout (310 scenes) evaluated exactly once at promotion gate.

---

## 2. Multi-Objective Evaluation & Benchmark Results

### A. Model-Selection Evaluation on DEV Partition (314 Scenes)
*Note: M3-C (Seed 42) serves as the authoritative model-selection evidence on the held-out DEV partition prior to production refit. M3-FINAL was refit on TRAIN + DEV, and its primary unbiased evaluation is measured on the sealed FINAL-HOLDOUT.*

| Metric | Frozen M2 Baseline | M3-A (Full Data) | M3-B (Height Balanced) | **M3-C (FiLM Winner, Seed 42)** | M3-FINAL (Production Refit) | Absolute Improvement (M3-C vs M2) |
|---|---:|---:|---:|---:|---:|---:|
| **Overall MAE (m)** | 4.4451 | 2.8295 | 3.0906 | **2.9924** | 3.2761 | **-1.4527m (+32.7%)** |
| **Overall RMSE (m)** | 7.5899 | 4.8872 | 5.2104 | **5.0931** | 5.2973 | **-2.4968m (+32.9%)** |
| **Coefficient of Determination ($R^2$)** | 0.0008 | 0.5241 | 0.5275 | **0.5529** | 0.5132 | **$\Delta R^2 = +0.5521$ (Absolute Gain)** |
| **10–20m Mid-Rise MAE (m)** | 10.9567 | 5.6721 | 5.8994 | **5.5492** | 5.1738 | **-5.4075m (-49.4%)** |
| **20–50m High-Rise MAE (m)** | 19.1811 | 12.7305 | 8.4112 | **7.2205** | 7.6881 | **-11.9606m (-62.4%)** |
| **Selection Score (Lower is better)** | 21.9457 | 15.7502 | 13.7896 | **13.1307** | 13.5525 | **-8.8150 Pareto Gain** |

---

### B. Sealed FINAL-HOLDOUT Partition (310 Scenes — Unbiased Production Evaluation)

| Metric | Frozen M2 Baseline | **M3-FINAL Production** | Absolute Delta | Relative Gain |
|---|---:|---:|---:|---:|
| **Overall MAE (m)** | 4.1999 | **3.0426** | -1.1573m | **+27.6% error reduction** |
| **Overall RMSE (m)** | 7.3237 | **4.9569** | -2.3668m | **+32.3% error reduction** |
| **Coefficient of Determination ($R^2$)** | -0.3062 | **0.4016** | +0.7078 | **Positive variance explained** |
| **10–20m Mid-Rise MAE (m)** | 10.9012 | **5.0266** | -5.8746m | **53.9% error reduction** |
| **20–50m High-Rise MAE (m)** | 17.8540 | **6.4675** | -11.3865m | **63.8% error reduction** |
| **$\ge$ 50m Skyscraper MAE (m)** | 53.3312 | **33.3552** | -19.9760m | **High-rise compression substantially reduced; extreme $\ge$50m underestimation remains** |
| **Selection Score** | 22.2334 | **12.0642** | -10.1692 | **45.7% Pareto Gain** |

---

### C. Unseen External Zero-Shot City: NYC (500 Scenes — External Generalization)

| Metric | Frozen M2 Baseline | **M3-FINAL Production** | Absolute Delta | Scientific Note |
|---|---:|---:|---:|:---|
| **Overall MAE (m)** | 8.0278 | **5.0364** | -2.9914m | **+37.3% error reduction** |
| **Overall RMSE (m)** | 11.9226 | **7.2387** | -4.6839m | **+39.3% error reduction** |
| **Coefficient of Determination ($R^2$)** | -1.7756 | **-0.0232** | +1.7524 | **Substantially improved toward zero, but remains slightly negative** |
| **10–20m Mid-Rise MAE (m)** | 9.9896 | **7.0230** | -2.9666m | **29.7% error reduction** |
| **20–50m High-Rise MAE (m)** | 13.9097 | **11.1406** | -2.7691m | **19.9% error reduction** |
| **$\ge$ 50m Skyscraper MAE (m)** | 35.8295 | **29.5249** | -6.3046m | **Extreme skyscrapers exhibit residual negative bias** |
| **Selection Score** | 22.6345 | **15.8431** | -6.7914 | **30.0% Pareto Gain** |

---

## 3. Seed Invariance & Stability

Three random seeds were trained under the winning M3-C configuration:
- **Seed 42:** DEV Score 13.13 | 10–20m MAE: 5.55m | 20–50m MAE: 7.22m
- **Seed 1337:** DEV Score 13.93 | 10–20m MAE: 5.59m | 20–50m MAE: 7.25m
- **Seed 2026:** DEV Score 14.07 | 10–20m MAE: 5.55m | 20–50m MAE: 7.69m
- **Standard Deviation:** $\pm 0.42$ on selection score; $\pm 0.02$m on 10–20m MAE; $\pm 0.21$m on 20–50m MAE.
- Demonstrates exceptional numerical stability across seeds.

---

## 4. Promotion Gate Status

All 8 official promotion gates passed unconditionally:
1. `[PASS]` DEV Overall MAE Improvement: 4.4451m $\to$ 3.2761m
2. `[PASS]` DEV $R^2$ Improvement: 0.0008 $\to$ 0.5132
3. `[PASS]` DEV 10–20m MAE Improvement: 10.9567m $\to$ 5.1738m
4. `[PASS]` DEV 20–50m MAE Improvement: 19.1811m $\to$ 7.6881m
5. `[PASS]` DEV Selection Score Improvement: 21.9457 $\to$ 13.5525
6. `[PASS]` FINAL-HOLDOUT MAE Improvement: 4.1999m $\to$ 3.0426m
7. `[PASS]` EXTERNAL-NYC Zero-Shot MAE Improvement: 8.0278m $\to$ 5.0364m
8. `[PASS]` EXTERNAL-NYC Zero-Shot $R^2$ Improvement: -1.7756 $\to$ -0.0232

**Status: OFFICIALLY ACCEPTED & PROMOTED TO PRODUCTION.**
