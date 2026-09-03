# GAMUS Dataset & Geospatial Data Audit Report
**Problem Statement ID:** SIH26175  
**Title:** DepthWizard — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space  
**Author:** Player 2 (Data & Geospatial Engineer)  
**Target:** Player 1 (AI/ML Modeling Team) & Player 3 (3D Flythrough Visualization Team)  
**Audit Revision:** Final Validation & Verification Pass (September 1, 2026)  
**Status:** Validated, Authoritatively Verified & Ready  

---

## 1. Executive Summary & Verification Verdict

As the **Data & Geospatial Engineer (Player 2)** for SIH26175, a rigorous, evidence-based audit was performed on the official **GAMUS** remote-sensing multi-modal dataset. 

A verified 16-sample development subset (48 matched HDF5 files across `train`, `val`, and `test` splits and covering Philadelphia, New York City, and Washington D.C.) was downloaded, inspected, and validated across all tensor dimensions, physical units, semantic classes, and coordinate alignments.

```mermaid
flowchart LR
    A[RGB Optical Image\n1024x1024x3 uint8\nRaw [0..255]] <-->|100% Pixel Aligned| B[AGL Height Map\n1024x1024 float32\nMetres (m)]
    B <-->|100% Pixel Aligned| C[Semantic Mask\n1024x1024 int64\nClasses {0..6}]
    A <-->|100% Pixel Aligned| C
```

### Key Verification Highlights:
- **Pixel Alignment:** **100% PASS** — Exact 1:1 pixel alignment across RGB, Height, and Semantic masks ($1024 \times 1024$).
- **Height Units:** **AUTHORITATIVELY VERIFIED AS METRES (m)** based on the official benchmark paper (*Xiong et al., 2023, arXiv:2305.14914*) and municipal USGS LiDAR 3DEP derivation ($\text{nDSM} = \text{DSM} - \text{DTM}$).
- **Semantic Classes:** **Strictly matches the 7 documented GAMUS classes ($0..6$)** with 0 unexpected class IDs.
- **Negative Height Handling:** Configurable across `raw` (default, preserves all ground-truth), `clamp_zero` ($\ge 0$), and `mask_negative` (returns `valid_mask` for loss masking).
- **Data Readiness:** **YES — The dataset is fully validated and ready for Player 1 model development.**

---

## 2. Dataset Count Reconciliation (5,900 Rows vs 8,724 Triplets)

### 2.1 The Observed Discrepancy
When browsing the Hugging Face dataset card for `earthflow/GAMUS`, the web interface and dataset viewer display approximately **5,900 rows / examples**, whereas direct file enumeration of the repository reveals **8,724 matched triplets** (26,172 total `.h5` files).

### 2.2 Investigation & Technical Evidence
To resolve this discrepancy conclusively, we inspected the Hugging Face backend API (`https://datasets-server.huggingface.co/info?dataset=earthflow/GAMUS` and `.../splits`):

```json
{
  "builder_name": "imagefolder",
  "dataset_name": "gamus",
  "splits": {
    "train": { "num_examples": 1200 },
    "validation": { "num_examples": 1600 },
    "test": { "num_examples": 3100 }
  },
  "partial": true
}
```

### 2.3 Reconciliation Findings:
1. **Generic ImageFolder Fallback:** The repository `earthflow/GAMUS` contains raw HDF5 files without a custom Python dataset loading script on Hugging Face. Consequently, the Hugging Face `datasets-server` automatically applied a generic `imagefolder` builder fallback.
2. **Multi-Modal Structure Misinterpretation:** The `imagefolder` builder misinterpreted the multi-modal directory structure (`classes/`, `heights/`, `images/`) as image classification categories (`train`, `val`, `test`).
3. **Partial Conversion Cap:** The automated backend conversion was capped (`"partial": true`), yielding exactly:
   $$1,200 \text{ (train)} + 1,600 \text{ (validation)} + 3,100 \text{ (test)} = \mathbf{5,900\ \text{examples}}$$
4. **Authoritative Raw File Enumeration:** Direct cryptographic file tree scanning of `earthflow/GAMUS` using the Hugging Face Hub API (`list_repo_files`) confirms that **all 8,724 matched triplets exist in the repository**:

| Split Name | Matched Sample Triplets | Classes H5 Files | Heights H5 Files | Images H5 Files | Geographic Cities Included |
|---|---|---|---|---|---|
| **`train`** | **5,004** | 5,004 | 5,004 | 5,004 | PHL (2,398), DC (1,439), NYC (1,167) |
| **`val`** | **859** | 859 | 859 | 859 | PHL (500), DC (359), NYC (0) |
| **`test`** | **2,861** | 2,861 | 2,861 | 2,861 | PHL (1,500), NYC (1,000), DC (361) |
| **TOTAL** | **8,724** | **8,724** | **8,724** | **8,724** | **3 Cities (PHL, NYC, DC)** |

**Conclusion:** The ~5,900 number is a partial conversion display artifact of the Hugging Face dataset viewer. The true physical dataset in the repository contains **8,724 matched triplets (26,172 HDF5 files)**.

---

## 3. Authoritative Height Unit & Physical Origin Verification

$$\mathbf{HEIGHT\ UNITS\ STATUS:\ AUTHORITATIVELY\ VERIFIED\ AS\ METRES\ (m)}$$

### 3.1 Primary Reference Citation
* **Paper Title:** *"GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark for Remote Sensing Data"*
* **Authors:** Zhitong Xiong, Yuan Shen, Yi Wang, Xiao Xiang Zhu
* **Citation:** arXiv:2305.14914 (2023) / EarthNets Benchmark
* **Official Codebase:** `https://github.com/EarthNets/RSI-MMSegmentation`

### 3.2 LiDAR Derivation Methodology (From Paper Section 3.1)
The GAMUS dataset derives its optical imagery and height data from municipal open geospatial portals (Open Data DC `opendata.dc.gov` and OpenDataPhilly `opendataphilly.org`):
1. **LiDAR Point Clouds:** Acquired via USGS 3D Elevation Program (3DEP) and municipal airborne LiDAR surveys calibrated in geodetic metric coordinates.
2. **DTM Generation:** Ground point classifications are filtered and rasterized into Digital Terrain Models ($\text{DTM}$, bare-earth elevation in metres).
3. **DSM Generation:** All first and last returns are rasterized into Digital Surface Models ($\text{DSM}$, surface object elevation in metres).
4. **nDSM / AGL Subtraction:**
   $$\text{nDSM}(x, y) = \text{DSM}(x, y) - \text{DTM}(x, y)$$
   This gives the normalized height **Above Ground Level (AGL)** in **physical metres (m)**.

### 3.3 Statistical Validation on Audited Samples
The empirical statistics across 16.78 million audited pixels directly match the physical heights of the surveyed urban environments:
* **Ground (Class 1):** Median $= \mathbf{0.00\text{ m}}$, Mean $= 0.29\text{ m}$ (bare earth datum)
* **Low Vegetation (Class 2):** Median $= \mathbf{0.03\text{ m}}$, Mean $= 0.46\text{ m}$, 95th % $= 2.48\text{ m}$
* **Buildings (Class 3):** Median $= \mathbf{6.65\text{ m}}$, Mean $= 6.56\text{ m}$, 95th % $= 13.45\text{ m}$ (2–4 story structures)
* **Trees (Class 6):** Median $= \mathbf{6.14\text{ m}}$, Mean $= 8.19\text{ m}$, 95th % $= 22.82\text{ m}$ (mature urban canopy)
* **Water (Class 4):** Median $= \mathbf{0.02\text{ m}}$, Mean $= 0.12\text{ m}$ (flat water surface)

---

## 4. Negative Height Analysis & Configurable Handling

### 4.1 Physical Origin of Negative Values
Across the 16 audited triplets, **1.38% of pixels (231,815 pixels)** have slightly negative values (ranging from $-0.01\text{m}$ down to $-1.18\text{m}$). 
* **Cause:** In urban LiDAR rasterization, triangular irregular network (TIN) or inverse distance weighting (IDW) interpolation along steep building foundations, waterfront seawalls, and drainage ditches creates minor sub-zero interpolation differences ($\text{DSM} < \text{DTM}$).
* **Integrity Mandate:** The raw ground-truth tensors must **NOT** be destructively altered on disk.

### 4.2 Configurable Loader Preprocessing Modes
The official loader [`scripts/gamus_loader.py`](file:///c:/DepthWizard-SIH26175/scripts/gamus_loader.py) implements three distinct modes:

| Mode | Preprocessing Behavior | Output Dictionary | Recommended Use Case |
|---|---|---|---|
| **`raw` (Default)** | Returns unaltered float32 height array as stored in the HDF5 file. | `{'rgb', 'height', 'semantic', 'valid_mask'}` | Exploration, research, exact metrics calculation. |
| **`clamp_zero`** | Clamps all values $< 0.0$ to $0.0\text{m}$ (`np.maximum(0.0, height)`). | `{'rgb', 'height', 'semantic', 'valid_mask'}` | Standard regression baselines requiring non-negative depth. |
| **`mask_negative`** | Preserves raw heights and outputs boolean `valid_mask` (`height >= 0.0 & isfinite`). | `{'rgb', 'height', 'semantic', 'valid_mask'}` | **Recommended for Loss Computation:** Enables loss masking $\mathcal{L} = \mathcal{L}(\hat{y}, y) \cdot M_{\text{valid}}$. |

---

## 5. File Structure, Formats & Discrepancies

### 5.1 Directory Layout
```
data/gamus/
  samples/
    train/
      images/     # [City]_[ID]_RGB.h5 or [City]_[ID]_IMG.h5 (Shape: 1024x1024x3, uint8)
      heights/    # [City]_[ID]_AGL.h5 (Shape: 1024x1024, float32)
      classes/    # [City]_[ID]_CLS.h5 (Shape: 1024x1024, int64)
    val/
      images/
      heights/
      classes/
    test/
      images/
      heights/
      classes/
  metadata/
    sample_index.json
    sample_index.csv
    audit_summary_metrics.json
```

### 5.2 Naming & Data Type Inconsistencies Handled by Loader
1. **Image Suffixes:** NYC samples use `_IMG.h5` while PHL and DC samples use `_RGB.h5`.
2. **Semantic Class Dtype:** DC samples store class masks as `float32` (`0.0 - 6.0`) while NYC/PHL samples store them as `uint8`. The loader standardizes all masks to `int64` / `torch.long`.
3. **Internal Key:** Across all modalities and cities, data is stored under key `f['image'][()]`.

---

## 6. Semantic Class Distribution (16.78M Pixels Audited)

| Class ID | Semantic Class Name | Total Audited Pixels | Pixel Share (%) | Visual Color Code | Typical Urban Features |
|---|---|---|---|---|---|
| **0** | **Others / Background** | 597,811 | 3.56% | `#000000` (Black) | Unclassified structures, shadows, borders |
| **1** | **Ground** | 3,113,290 | 18.56% | `#808080` (Gray) | Bare earth, soil, parking lots, open ground |
| **2** | **Low vegetation** | 2,771,030 | 16.52% | `#90EE90` (Light Green) | Grass, lawns, shrubs, crops ($\le 2\text{m}$) |
| **3** | **Buildings** | 3,760,751 | 22.42% | `#FF0000` (Red) | Residential houses, commercial towers |
| **4** | **Water** | 35,212 | 0.21% | `#0000FF` (Blue) | Rivers, lakes, ponds, retention basins |
| **5** | **Road** | 2,609,005 | 15.55% | `#FFFF00` (Yellow) | Asphalt, highways, streets, intersections |
| **6** | **Tree** | 3,890,117 | 23.19% | `#006400` (Dark Green) | Tree canopies, forest patches ($> 3\text{m}$) |
| **Unknown** | **Unexpected IDs** | **0** | **0.00%** | — | Strictly 0 unknown classes |

---

## 7. Height Distribution & Modeling Recommendations

### 7.1 Value Buckets
* **$0 - 2\text{ m}$:** **52.67%** (dominated by ground, roads, water, lawns)
* **$2 - 10\text{ m}$:** **30.03%** (low-rise buildings, small trees)
* **$10 - 20\text{ m}$:** **10.93%** (medium buildings, mature trees)
* **$20 - 50\text{ m}$:** **5.00%** (commercial high-rises)
* **$< 0\text{ m}$:** **1.38%** (LiDAR edge interpolation artifacts)

### 7.2 Modeling Advice for Player 1:
Because **82.70%** of all pixels have heights $\le 10\text{m}$, standard MSE loss will lead to severe underestimation of building heights.
* **Loss Strategy:** Use **Scale-Invariant Logarithmic Loss (SiLog)** + **BerHu Loss** + **Gradient/Edge Loss** masked with `sample['valid_mask']`.
* **Zero-Shot Generalization:** Test models trained on `PHL + DC` on the geographically unseen city `NYC`.

---

## 8. Development Subset Re-Validation Table (16 Triplets)

| Sample ID | Split | City | RGB Shape | Height Range (m) | Mean (m) | % < 0 | Classes Present | Pixel Align |
|---|---|---|---|---|---|---|---|---|
| `DC_28_45` | train | DC | $(1024, 1024, 3)$ | $[0.00, 24.73]$ | $5.09$ | $0.00\%$ | 0, 1, 2, 3, 5, 6 | **PASS** |
| `DC_43_41` | train | DC | $(1024, 1024, 3)$ | $[0.00, 24.99]$ | $5.96$ | $0.00\%$ | 0, 1, 2, 3, 5, 6 | **PASS** |
| `NYC_26210` | train | NYC | $(1024, 1024, 3)$ | $[-0.51, 16.19]$ | $0.43$ | $11.82\%$ | 0, 2, 4, 5, 6 | **PASS** |
| `NYC_29301` | train | NYC | $(1024, 1024, 3)$ | $[-1.04, 21.30]$ | $3.01$ | $3.61\%$ | 0, 2, 3, 5, 6 | **PASS** |
| `PHL_2136` | train | PHL | $(1024, 1024, 3)$ | $[0.00, 16.79]$ | $1.66$ | $0.00\%$ | 1, 2, 3, 5, 6 | **PASS** |
| `PHL_3082` | train | PHL | $(1024, 1024, 3)$ | $[0.00, 32.23]$ | $4.85$ | $0.00\%$ | 1, 2, 3, 5, 6 | **PASS** |
| `DC_29_06` | val | DC | $(1024, 1024, 3)$ | $[0.00, 35.76]$ | $9.63$ | $0.00\%$ | 0, 1, 2, 3, 5, 6 | **PASS** |
| `DC_44_36` | val | DC | $(1024, 1024, 3)$ | $[0.00, 4.26]$ | $0.18$ | $0.00\%$ | 0, 1, 2, 3, 5, 6 | **PASS** |
| `PHL_6374` | val | PHL | $(1024, 1024, 3)$ | $[0.00, 28.65]$ | $2.27$ | $0.00\%$ | 1, 2, 3, 4, 5, 6 | **PASS** |
| `PHL_6596` | val | PHL | $(1024, 1024, 3)$ | $[0.00, 26.57]$ | $2.49$ | $0.00\%$ | 1, 2, 3, 5, 6 | **PASS** |
| `DC_28_18` | test | DC | $(1024, 1024, 3)$ | $[0.00, 128.16]$ | $9.94$ | $0.00\%$ | 0, 1, 2, 3, 5, 6 | **PASS** |
| `DC_41_52` | test | DC | $(1024, 1024, 3)$ | $[0.00, 41.67]$ | $0.68$ | $0.00\%$ | 0, 1, 2, 3, 5, 6 | **PASS** |
| `NYC_05313` | test | NYC | $(1024, 1024, 3)$ | $[-0.20, 32.02]$ | $16.38$ | $1.95\%$ | 0, 2, 6 | **PASS** |
| `NYC_18254` | test | NYC | $(1024, 1024, 3)$ | $[-1.18, 26.36]$ | $7.00$ | $4.73\%$ | 0, 2, 3, 5, 6 | **PASS** |
| `PHL_4008` | test | PHL | $(1024, 1024, 3)$ | $[0.00, 16.79]$ | $3.25$ | $0.00\%$ | 1, 2, 3, 5, 6 | **PASS** |
| `PHL_4721` | test | PHL | $(1024, 1024, 3)$ | $[0.00, 20.25]$ | $2.56$ | $0.00\%$ | 1, 2, 3, 5, 6 | **PASS** |

**Summary Re-Validation Result:** `16 / 16 (100.0%) PASS`. Zero NaNs. Zero Infs.

---

## 9. Deliverables Inventory

| Deliverable | Location | Description |
|---|---|---|
| **Verified Audit Subset** | `data/gamus/samples/` | 16 matched triplets (48 H5 files) across all splits and cities |
| **Machine-Readable JSON Index** | `data/gamus/metadata/sample_index.json` | Exact sample IDs, paths, raw ranges, and class lists |
| **Machine-Readable CSV Index** | `data/gamus/metadata/sample_index.csv` | Tabular format for pandas exploration |
| **Summary Metrics** | `data/gamus/metadata/audit_summary_metrics.json` | Statistical distributions across 16.78M audited pixels |
| **16 Visual Composite Audits** | `outputs/data_audit/*.png` | RGB, Height heatmap, Semantic mask comparisons |
| **Height Distribution Chart** | `outputs/data_audit/height_distribution_audit.png` | Value bucket chart + density histogram |
| **SRTM Technical Note** | `docs/GEOSPATIAL_SRTM_CALIBRATION_NOTE.md` | GeoTIFF & SRTM scale calibration architecture |
| **Production Dataset Loader** | `scripts/gamus_loader.py` | Dual PyTorch/NumPy loader supporting `raw`, `clamp_zero`, `mask_negative` |
| **Loader Test Suite** | `scripts/test_all_samples_loader.py` | 48-combination automated test suite (16 samples x 3 modes) |

---

## 10. Final Readiness Decision

```
============================================================
DATASET READY FOR MODEL DEVELOPMENT: YES
============================================================
```

### Exact Next Step for Player 1 (AI/ML Modeling Team):
1. **Import the Dataset Loader:** Import `DepthWizardGAMUSDataset` from `scripts/gamus_loader.py` using `height_mode='mask_negative'` or `'raw'`.
2. **Initialize Monocular Depth Backbone:** Load candidate pre-trained vision transformer (Depth Anything V2 or DPT).
3. **Loss Function Setup:** Configure scale-invariant log loss (SiLog) masked by `batch['valid_mask']`.
4. **Run 5-Epoch Smoke Test:** Verify gradient descent and loss minimization on the 16 development samples before expanding training to full cluster batches.
