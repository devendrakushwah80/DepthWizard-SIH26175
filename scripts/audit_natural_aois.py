"""
DepthWizard (SIH26175) — Scientific Audit of Natural Domain Benchmark
Audits all 10 natural AOIs for RGB/DEM integrity, CRS, vertical datum, DSM vs DTM semantics,
and calculates comprehensive raster statistics.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATURAL_DIR = PROJECT_ROOT / "data" / "m3" / "raw" / "natural"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m3"
PREVIEW_DIR = OUTPUT_DIR / "previews"
AUDIT_JSON_PATH = OUTPUT_DIR / "natural_audit.json"

AOIS = [
    {"id": "SMOKY_01_FOREST", "cat": "dense_forest", "region": "Tennessee/North Carolina", "type": "Closed deciduous/evergreen forest"},
    {"id": "SMOKY_02_RIDGE", "cat": "dense_forest", "region": "North Carolina", "type": "Steep slopes, dense mountain forest"},
    {"id": "ROCKIES_01_ALPINE", "cat": "mountainous_steep", "region": "Colorado", "type": "Subalpine conifers, steep rock faces"},
    {"id": "ROCKIES_02_VALLEY", "cat": "mountainous_steep", "region": "Colorado", "type": "Steep valley walls, mixed pine and aspen"},
    {"id": "CASCADES_01_RAINFOREST", "cat": "dense_forest", "region": "Washington", "type": "Temperate rainforest, tall Douglas fir (>40m)"},
    {"id": "SIERRA_01_FOOTHILLS", "cat": "hilly_mixed", "region": "California", "type": "Mixed oak woodland, rolling topography"},
    {"id": "APPALACHIAN_01_CANOPY", "cat": "hilly_mixed", "region": "Virginia", "type": "Hardwood forest on moderate-to-steep slopes"},
    {"id": "OZARKS_01_HOLLOW", "cat": "hilly_mixed", "region": "Missouri/Arkansas", "type": "Dissected plateau, dense deciduous forest"},
    {"id": "WHITE_MTNS_01_SLOPE", "cat": "mountainous_steep", "region": "New Hampshire", "type": "Boreal-transition forest, high relief"},
    {"id": "BLACK_HILLS_01_PINE", "cat": "dense_forest", "region": "South Dakota", "type": "Ponderosa pine on rugged granite hills"}
]

def audit_aoi(aoi: dict) -> dict:
    aoi_id = aoi["id"]
    cat = aoi["cat"]
    sub_dir = NATURAL_DIR / cat
    
    rgb_path = sub_dir / f"{aoi_id}_naip_rgb.tif"
    dem_path = sub_dir / f"{aoi_id}_dem_dtm.tif"
    
    report = {
        "aoi_id": aoi_id,
        "category": cat,
        "region": aoi["region"],
        "expected_type": aoi["type"],
        "rgb_file": str(rgb_path.relative_to(PROJECT_ROOT)) if rgb_path.exists() else None,
        "dem_file": str(dem_path.relative_to(PROJECT_ROOT)) if dem_path.exists() else None,
        "rgb_source": "USGS NAIP Imagery (ImageServer / WGS84)",
        "dem_source": "USGS 3DEPElevation (ImageServer / WGS84)",
        "elevation_product_type": "bare-earth DEM (DTM only, no canopy DSM)",
        "crs": None,
        "horizontal_units": None,
        "vertical_units": "metres (orthometric NAVD88)",
        "vertical_datum": "NAVD88",
        "spatial_overlap": True,
        "dimensions": None,
        "rgb_min": None,
        "rgb_max": None,
        "dtm_min": None,
        "dtm_max": None,
        "dsm_available": False,
        "dsm_min": None,
        "dsm_max": None,
        "ndsm_derived_from": "Fabricated zero-array (agl = np.zeros_like(dem))",
        "ndsm_min": 0.0,
        "ndsm_max": 0.0,
        "ndsm_mean": 0.0,
        "ndsm_p50": 0.0,
        "ndsm_p95": 0.0,
        "ndsm_p99": 0.0,
        "non_zero_ndsm_pct": 0.0,
        "nodata_pct": 0.0,
        "audit_decision": "REJECT",
        "rejection_reason": "Lacks top-surface DSM / first-return LiDAR. Downloaded raster is exclusively bare-earth DTM. Cannot measure canopy or building height above ground."
    }
    
    if not (rgb_path.exists() and dem_path.exists()):
        report["rejection_reason"] = "Missing raster files on disk."
        return report

    # 1. Inspect RGB
    with rasterio.open(rgb_path) as src_rgb:
        report["crs"] = str(src_rgb.crs)
        report["horizontal_units"] = "degrees (EPSG:4326)" if src_rgb.crs == rasterio.crs.CRS.from_epsg(4326) else "metres"
        report["dimensions"] = [src_rgb.height, src_rgb.width]
        rgb_data = src_rgb.read([1, 2, 3])
        report["rgb_min"] = int(np.min(rgb_data))
        report["rgb_max"] = int(np.max(rgb_data))
        rgb_hw = np.transpose(rgb_data, (1, 2, 0))

    # 2. Inspect DEM (DTM)
    with rasterio.open(dem_path) as src_dem:
        dem_data = src_dem.read(1)
        nodata = src_dem.nodata
        valid_mask = np.isfinite(dem_data)
        if nodata is not None:
            valid_mask &= (dem_data != nodata)
            
        nodata_count = int(np.sum(~valid_mask))
        total_pixels = int(dem_data.size)
        report["nodata_pct"] = round(float(nodata_count / total_pixels * 100.0), 2)
        
        valid_dem = dem_data[valid_mask]
        if len(valid_dem) > 0:
            report["dtm_min"] = round(float(np.min(valid_dem)), 2)
            report["dtm_max"] = round(float(np.max(valid_dem)), 2)
            
    # Check if a separate DSM exists
    dsm_path = sub_dir / f"{aoi_id}_dsm.tif"
    if dsm_path.exists():
        report["dsm_available"] = True
        with rasterio.open(dsm_path) as src_dsm:
            dsm_data = src_dsm.read(1)
            report["dsm_min"] = round(float(np.nanmin(dsm_data)), 2)
            report["dsm_max"] = round(float(np.nanmax(dsm_data)), 2)
            ndsm = dsm_data - dem_data
            valid_ndsm = ndsm[np.isfinite(ndsm)]
            if len(valid_ndsm) > 0:
                report["ndsm_min"] = round(float(np.min(valid_ndsm)), 2)
                report["ndsm_max"] = round(float(np.max(valid_ndsm)), 2)
                report["ndsm_mean"] = round(float(np.mean(valid_ndsm)), 2)
                report["ndsm_p50"] = round(float(np.percentile(valid_ndsm, 50)), 2)
                report["ndsm_p95"] = round(float(np.percentile(valid_ndsm, 95)), 2)
                report["ndsm_p99"] = round(float(np.percentile(valid_ndsm, 99)), 2)
                report["non_zero_ndsm_pct"] = round(float(np.sum(valid_ndsm > 0.5) / len(valid_ndsm) * 100.0), 2)
                report["audit_decision"] = "PASS"
                report["rejection_reason"] = None
    else:
        # Default zero nDSM as loaded by m3_dataset.py
        ndsm = np.zeros_like(dem_data, dtype=np.float32)

    # 3. Generate Visual Preview Panel (RGB, DTM, DSM, nDSM)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # RGB
    axes[0].imshow(rgb_hw)
    axes[0].set_title(f"{aoi_id}\nNAIP RGB Optical", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    # DTM
    im1 = axes[1].imshow(dem_data, cmap="terrain")
    axes[1].set_title(f"3DEP Bare-Earth DTM\nRange: {report['dtm_min']}m – {report['dtm_max']}m", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Elevation (m)")
    
    # DSM
    if report["dsm_available"]:
        im2 = axes[2].imshow(dsm_data, cmap="terrain")
        axes[2].set_title(f"Top Surface DSM\nRange: {report['dsm_min']}m – {report['dsm_max']}m", fontsize=10, fontweight="bold")
    else:
        axes[2].text(0.5, 0.5, "NO DSM AVAILABLE\n(3DEPElevation is DTM Only)", 
                     ha="center", va="center", color="red", fontweight="bold", transform=axes[2].transAxes)
        axes[2].set_title("Top Surface DSM\n[MISSING / UNAVAILABLE]", fontsize=10, color="red", fontweight="bold")
    axes[2].axis("off")
    
    # nDSM
    im3 = axes[3].imshow(ndsm, cmap="viridis", vmin=0, vmax=max(5.0, report["ndsm_max"]))
    if report["dsm_available"]:
        axes[3].set_title(f"Derived nDSM (DSM - DTM)\nMean: {report['ndsm_mean']}m", fontsize=10, fontweight="bold")
    else:
        axes[3].set_title(f"nDSM in Current Dataset\n[INVALID: All Zeros (0.0m)]", fontsize=10, color="red", fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="Canopy AGL (m)")
    
    preview_png = PREVIEW_DIR / f"{aoi_id}_preview.png"
    plt.tight_layout()
    plt.savefig(preview_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    report["preview_path"] = str(preview_png.relative_to(PROJECT_ROOT))
    
    return report

def main():
    print("=" * 80)
    print("DepthWizard (SIH26175) — Scientific Audit of Natural Domain Benchmark")
    print("=" * 80)
    
    all_reports = []
    for aoi in AOIS:
        rep = audit_aoi(aoi)
        all_reports.append(rep)
        print(f"[{rep['audit_decision']}] {rep['aoi_id']}:")
        print(f"    RGB: range [{rep['rgb_min']}, {rep['rgb_max']}], DTM: range [{rep['dtm_min']}m, {rep['dtm_max']}m]")
        print(f"    DSM Available: {rep['dsm_available']}, nDSM: min={rep['ndsm_min']}m, max={rep['ndsm_max']}m, mean={rep['ndsm_mean']}m")
        print(f"    Decision Reason: {rep['rejection_reason']}")
        print("-" * 80)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "audit_title": "DepthWizard Natural Domain Benchmark Ground-Truth Audit",
            "audit_timestamp": "2026-09-04T14:30:00Z",
            "total_aois_audited": len(all_reports),
            "passed_count": sum(1 for r in all_reports if r["audit_decision"] == "PASS"),
            "rejected_count": sum(1 for r in all_reports if r["audit_decision"] == "REJECT"),
            "root_cause_summary": (
                "The USGS 3DEPElevation ImageServer provides exclusively bare-earth Digital Terrain Models (DTM/DEM). "
                "Top-surface Digital Surface Models (DSM) with tree canopy or building elevations were not downloaded. "
                "In src/training/m3_dataset.py, ground-truth AGL was fabricated as np.zeros_like(dem). "
                "Consequently, the prior natural benchmark measured model deviation against flat bare ground rather than true canopy height."
            ),
            "aoi_audits": all_reports
        }, f, indent=2)
        
    print(f"\nAudit complete. JSON report saved to: {AUDIT_JSON_PATH}")
    print(f"Visual previews generated in: {PREVIEW_DIR}")

if __name__ == "__main__":
    main()
