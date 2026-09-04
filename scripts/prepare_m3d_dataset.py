"""
DepthWizard (SIH26175) — M3-D Natural Training Set Builder
Selects 300 unexposed US3D scenes with real airborne LiDAR-derived nDSM
for mixed-domain natural training (zero overlap with Holdout, NYC, or Natural Benchmark).
"""

from __future__ import annotations

import os
import glob
import json
from pathlib import Path
import rasterio
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
US3D_DIR = PROJECT_ROOT / "data" / "m3" / "raw" / "us3d"
MANIFEST_DIR = PROJECT_ROOT / "data" / "m3" / "manifests"

def build_m3d_splits():
    print("=" * 70)
    print("Building M3-D Mixed-Domain Training Set")
    print("=" * 70)

    # 1. Gather all existing excluded IDs
    excluded_ids = set()
    for fname in ["dev.txt", "final_holdout.txt", "nyc_external.txt"]:
        fpath = MANIFEST_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    sid = line.strip().replace("US3D_", "").replace("GAMUS_", "")
                    excluded_ids.add(sid)

    # Add the 90 tiles from the validated natural benchmark
    val_json = PROJECT_ROOT / "outputs" / "m3" / "VALIDATED_NATURAL_METRICS.json"
    # Or find from the selection logic
    from scripts.evaluate_validated_natural import select_validated_natural_tiles
    val_tiles = select_validated_natural_tiles()
    for group in val_tiles.values():
        for item in group:
            excluded_ids.add(item["id"])

    print(f"Total excluded IDs (DEV, Holdout, NYC, Natural Benchmark): {len(excluded_ids)}")

    # 2. Gather all US3D candidate tiles
    all_ndsm = sorted(glob.glob(str(US3D_DIR / "ndsm" / "*.tif")))
    candidate_tiles = []
    for p in all_ndsm:
        stem = Path(p).stem
        if stem not in excluded_ids:
            opt_p = US3D_DIR / "opt" / f"{stem}.tif"
            if opt_p.exists():
                candidate_tiles.append((p, str(opt_p), stem))

    print(f"Available unexposed candidate tiles: {len(candidate_tiles)}")

    # 3. Classify into strata
    forest_candidates = []
    mixed_candidates = []
    open_candidates = []

    for ndsm_p, opt_p, sid in candidate_tiles:
        with rasterio.open(ndsm_p) as src:
            d = src.read(1)
            valid = d[np.isfinite(d)]
            if len(valid) == 0:
                continue
            canopy_pct = float(np.mean((valid >= 4.0) & (valid <= 35.0)))
            mean_h = float(np.mean(valid))
            max_h = float(np.max(valid))

            item = {
                "record_id": f"US3D_{sid}",
                "dataset": "US3D_NATURAL",
                "city": "JAX",
                "region": "Florida USA",
                "rgb_path": opt_p,
                "agl_path": ndsm_p,
                "semantic_path": None,
                "dem_path": None,
                "gsd_m": 0.5,
                "gsd_known": True,
                "crs": "EPSG:32617",
                "height_unit_original": "metres",
                "height_unit_internal": "metres",
                "width": 512,
                "height": 512,
                "quality_tier": "TIER_A",
                "geographic_group": f"NATURAL_{sid[:7]}",
                "overlap_group": f"NATURAL_{sid[:7]}",
                "canopy_pct": canopy_pct,
                "mean_h": mean_h,
                "max_h": max_h,
            }

            if canopy_pct > 0.40 and mean_h > 4.0 and len(forest_candidates) < 100:
                forest_candidates.append(item)
            elif 0.15 < canopy_pct <= 0.40 and 2.0 < mean_h <= 4.0 and len(mixed_candidates) < 150:
                mixed_candidates.append(item)
            elif canopy_pct < 0.05 and mean_h < 1.0 and len(open_candidates) < 50:
                open_candidates.append(item)

        if len(forest_candidates) >= 100 and len(mixed_candidates) >= 150 and len(open_candidates) >= 50:
            break

    selected_natural = forest_candidates + mixed_candidates + open_candidates
    print(f"Selected {len(forest_candidates)} forest, {len(mixed_candidates)} mixed, {len(open_candidates)} open natural scenes.")
    print(f"Total new natural training scenes: {len(selected_natural)}")

    # 4. Combine with existing train.txt
    existing_train_ids = []
    with open(MANIFEST_DIR / "train.txt") as f:
        existing_train_ids = [line.strip() for line in f if line.strip()]

    new_natural_ids = [r["record_id"] for r in selected_natural]
    m3d_train_ids = existing_train_ids + new_natural_ids

    m3d_train_file = MANIFEST_DIR / "m3d_train.txt"
    with open(m3d_train_file, "w") as f:
        for sid in m3d_train_ids:
            f.write(f"{sid}\n")

    print(f"Saved M3-D train IDs to: {m3d_train_file} (Total: {len(m3d_train_ids)} scenes)")

    # 5. Update all_records.jsonl to include new natural records
    existing_records = []
    with open(MANIFEST_DIR / "all_records.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            existing_records.append(json.loads(line))

    existing_rec_ids = {r["record_id"] for r in existing_records}
    added_count = 0
    for r in selected_natural:
        if r["record_id"] not in existing_rec_ids:
            # Clean dictionary for JSON
            rec_clean = {k: v for k, v in r.items() if k not in ("canopy_pct", "mean_h", "max_h")}
            existing_records.append(rec_clean)
            added_count += 1

    with open(MANIFEST_DIR / "all_records.jsonl", "w", encoding="utf-8") as f:
        for r in existing_records:
            f.write(json.dumps(r) + "\n")

    print(f"Updated all_records.jsonl: added {added_count} natural records (Total: {len(existing_records)})")
    print("=" * 70)


if __name__ == "__main__":
    build_m3d_splits()
