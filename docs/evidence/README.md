# Public Evaluation Evidence

This directory contains compact machine-readable summaries copied from the
sealed local evaluation outputs during public-release preparation. Large raster
predictions, checkpoints, plots, databases, generated scenes, and raw datasets
remain local and are excluded by `.gitignore`.

The reports record historical experimental or validation runs; they are not
recomputed when the repository is cloned. Use the scripts and deterministic
manifests in the repository to reproduce applicable experiments with separately
obtained datasets.

Files are grouped by purpose:

- `stage_a2_*`: final-development selection and three-seed stability
- `nyc_*`: held-out NYC evaluation and leakage checks
- `geometry_*`: mesh-resolution and geometry checks
- `registration_*`: raster-to-mesh spatial-registration audit
- `product_*`: backend, geospatial, UI, and end-to-end product validation

The authoritative production checkpoint identity is maintained separately in
`models/m2_final/model_manifest.json` and `models/m2_final/checkpoint.sha256`.
