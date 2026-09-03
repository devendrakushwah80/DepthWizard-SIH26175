# Model Card — DepthWizard M2-FINAL

**Model identifier:** `M2-FINAL`  
**Architecture:** RDAHNetCore with frozen Depth Anything V2 Small prior  
**Task:** optical RGB to dense predicted AGL/nDSM in metres  
**Checkpoint:** `models/m2_final/M2_FINAL.pth`

## Model identity

- Selected candidate: trial 10
- Selected seed: 1337
- Selected epoch: 10
- Trainable parameters: 5,372,047
- Output parameterization: non-negative `softplus`
- Frozen prior: `depth-anything/Depth-Anything-V2-Small-hf`
- SHA256: `6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f`

The model was selected using the final development protocol without using NYC
for selection, tuning, or calibration. NYC was evaluated only after the model
was frozen.

## Recorded evaluation

| Evaluation | Tiles | Pixels | MAE (m) | RMSE (m) | Pearson r | R2 | Bias (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Final PHL/DC validation | 200 | ~200M | 2.518 | 4.738 | 0.806 | 0.631 | -0.918 |
| Held-out NYC (all pixels) | 496 | 478,021,075 | 4.844 | 8.144 | 0.272 | -0.222 | -3.141 |
| Held-out NYC (building pixels) | 496 | 66,797,828 | 4.388 | 6.415 | 0.335 | -0.215 | -0.423 |

### Held-Out NYC Height Stratification (Zero-Shot)

*NYC was strictly held out and never used for training, hyperparameter search, or calibration.*

| Height Range | Support | Pixel Count | MAE (m) | RMSE (m) | Bias (m) | Accuracy Threshold |
|---|---:|---:|---:|---:|---:|---|
| **0–2 m (Ground / Low)** | 50.96% | 243,616,087 | 0.985 | 2.672 | +0.771 | 82.49% within 1m, 89.19% within 2m |
| **2–10 m (Low-Rise)** | 25.60% | 122,385,904 | 4.425 | 5.294 | -1.745 | 61.22% within 5m, 97.86% within 10m |
| **10–20 m (Mid-Rise)** | 16.97% | 81,118,423 | 11.246 | 12.425 | -10.923 | Underestimation onset |
| **>20 m (High-Rise)** | 6.46% | 30,900,661 | 24.364 | 27.094 | -24.237 | Severe upper-tail compression |

The compact source reports are retained under `docs/evidence/`. Dataset manifests
are retained under `data/gamus/splits/`; raw imagery and full prediction arrays
are intentionally excluded from Git.

## Intended use

- rapid urban height-surface estimation from overhead RGB imagery
- exploratory emergency-response and planning visualization
- research comparisons and interactive 3D inspection

## Output interpretation

The predicted surface is AGL/nDSM. A georeferenced optical image alone does not
make it absolute elevation. An aligned terrain DEM with identical CRS, affine
transform, resolution, and raster shape is required to compute:

```text
absolute DSM = terrain DEM + predicted AGL
```

For non-georeferenced PNG/JPG inputs, the application does not invent horizontal
metric scale. A user-supplied GSD enables approximate horizontal measurements;
without it, horizontal units remain pixels.

## Limitations

- Single-view monocular estimation is geometrically ambiguous and cannot recover
  occluded structure with survey-grade fidelity.
- Tall-building predictions show upper-tail compression and underestimation.
- Training/evaluation evidence is primarily urban GAMUS imagery; mountainous and
  dense wild-forest performance is not established.
- An RGB-visible mountain is not evidence that predicted AGL represents absolute
  terrain topography.
- This model is not a LiDAR, stereo, InSAR, or survey-grade DEM replacement.

See `docs/KNOWN_LIMITATIONS.md` for the full limitations register.

## Integrity

The exact architecture and final validation values are recorded in
`models/m2_final/model_manifest.json`. The backend rejects a checkpoint whose
SHA256 differs from the sealed value above.
