"""
DepthWizard (SIH26175) — Unified Multi-Domain Dataset Manifest Builder
Author: DepthWizard Phase 2 Pipeline
Consolidates GAMUS, US3D (JAX/OMA), and Natural Domain (NAIP+3DEP) into a single canonical index.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

MANIFEST_JSONL = Path("data/m3/manifests/all_records.jsonl")
MANIFEST_PARQUET = Path("data/m3/manifests/all_records.parquet")


def compute_fast_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read first 1MB and last 1MB for fast reliable file fingerprinting
        head = f.read(1024 * 1024)
        h.update(head)
        f.seek(max(0, path.stat().st_size - 1024 * 1024))
        tail = f.read(1024 * 1024)
        h.update(tail)
        h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


def scan_gamus() -> list[dict]:
    print("Scanning local GAMUS dataset...")
    base = Path("data/gamus/dataset")
    records = []

    # Map all heights and classes by sample_id
    agl_map = {}
    cls_map = {}
    for p in base.glob("heights/**/*.h5"):
        sid = p.name.replace("_AGL.h5", "")
        agl_map[sid] = p
    for p in base.glob("classes/**/*.h5"):
        sid = p.name.replace("_CLS.h5", "")
        cls_map[sid] = p

    for img_p in base.glob("images/**/*.h5"):
        fname = img_p.name
        if fname.endswith("_RGB.h5"):
            sid = fname[:-7]
        elif fname.endswith("_IMG.h5"):
            sid = fname[:-7]
        else:
            continue

        city = sid.split("_")[0]
        agl_p = agl_map.get(sid)
        cls_p = cls_map.get(sid)

        # Invariant: Must have matching height map
        if not agl_p or not agl_p.exists() or agl_p.stat().st_size == 0:
            continue

        # Geographic grouping:
        # PHL_28_45 -> group is city + major block
        parts = sid.split("_")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            # Block group (e.g. 4x4 tiles per super-block to ensure zero boundary leakage)
            r = int(parts[1]) // 2
            c = int(parts[2]) // 2
            geo_group = f"{city}_B{r}_{c}"
        else:
            geo_group = sid

        records.append({
            "record_id": f"GAMUS_{sid}",
            "dataset": "GAMUS",
            "city": city,
            "region": "Mid-Atlantic USA",
            "rgb_path": str(img_p.resolve()),
            "agl_path": str(agl_p.resolve()),
            "semantic_path": str(cls_p.resolve()) if cls_p and cls_p.exists() else None,
            "dem_path": None,
            "gsd_m": 0.5,
            "gsd_known": True,
            "crs": "EPSG:32618" if city in ("DC", "PHL", "NYC") else None,
            "height_unit_original": "metres",
            "height_unit_internal": "metres",
            "width": 1024,
            "height": 1024,
            "quality_tier": "TIER_A",
            "geographic_group": geo_group,
            "overlap_group": geo_group,
            "split": "EXTERNAL_TEST" if city == "NYC" else "CANDIDATE",
            "checksum": compute_fast_sha256(img_p)
        })

    print(f"  Valid GAMUS records indexed: {len(records)} (DC/PHL/NYC)")
    return records


def scan_us3d() -> list[dict]:
    print("Scanning US3D dataset...")
    opt_dir = Path("data/m3/raw/us3d/opt")
    ndsm_dir = Path("data/m3/raw/us3d/ndsm")
    records = []

    if not opt_dir.exists() or not ndsm_dir.exists():
        print("  US3D directory not present yet.")
        return records

    for opt_p in opt_dir.glob("*.tif"):
        tile_id = opt_p.stem
        ndsm_p = ndsm_dir / f"{tile_id}.tif"
        if not (ndsm_p.exists() and ndsm_p.stat().st_size > 0):
            continue

        city = tile_id.split("_")[0]  # JAX or OMA
        parts = tile_id.split("_")
        # JAX_004_006_0_0 -> super-block group
        if len(parts) >= 3:
            geo_group = f"{city}_{parts[1]}_{parts[2]}"
        else:
            geo_group = tile_id

        records.append({
            "record_id": f"US3D_{tile_id}",
            "dataset": "US3D_DFC19",
            "city": city,
            "region": "Florida/Nebraska USA",
            "rgb_path": str(opt_p.resolve()),
            "agl_path": str(ndsm_p.resolve()),
            "semantic_path": None,
            "dem_path": None,
            "gsd_m": 0.5,
            "gsd_known": True,
            "crs": "EPSG:32617" if city == "JAX" else "EPSG:32614",
            "height_unit_original": "metres",
            "height_unit_internal": "metres",
            "width": 512,
            "height": 512,
            "quality_tier": "TIER_A",
            "geographic_group": geo_group,
            "overlap_group": geo_group,
            "split": "CANDIDATE",
            "checksum": compute_fast_sha256(opt_p)
        })

    print(f"  Valid US3D records indexed: {len(records)}")
    return records


def scan_natural() -> list[dict]:
    print("Scanning Natural Domain Benchmark dataset...")
    manifest_p = Path("data/m3/manifests/natural_domain_manifest.json")
    records = []

    if not manifest_p.exists():
        print("  Natural manifest not found.")
        return records

    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if item.get("status") != "VALID":
            continue
        naip_p = Path(item["naip_path"])
        dem_p = Path(item["dem_path"])
        if not (naip_p.exists() and dem_p.exists()):
            continue

        records.append({
            "record_id": f"NATURAL_{item['aoi_id']}",
            "dataset": f"NATURAL_{item['category'].upper()}",
            "city": item["aoi_id"],
            "region": item["region"],
            "rgb_path": str(naip_p.resolve()),
            "agl_path": None,  # Bare terrain; object height is near 0 except vegetation
            "semantic_path": None,
            "dem_path": str(dem_p.resolve()),
            "gsd_m": 1.0,
            "gsd_known": True,
            "crs": "EPSG:4326",
            "height_unit_original": "metres",
            "height_unit_internal": "metres",
            "width": 1024,
            "height": 1024,
            "quality_tier": "TIER_A",
            "geographic_group": item["aoi_id"],
            "overlap_group": item["aoi_id"],
            "split": "NATURAL_BENCHMARK",
            "checksum": compute_fast_sha256(naip_p)
        })

    print(f"  Valid Natural Domain records indexed: {len(records)}")
    return records


def main():
    print("=" * 70)
    print("DepthWizard SIH26175 — Building Canonical Multi-Domain Corpus Manifest")
    print("=" * 70)

    all_records = []
    all_records.extend(scan_gamus())
    all_records.extend(scan_us3d())
    all_records.extend(scan_natural())

    print(f"\nTotal multi-domain records indexed: {len(all_records)}")

    # Save JSONL
    MANIFEST_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSONL, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved canonical JSONL: {MANIFEST_JSONL}")

    # Save Parquet or CSV
    try:
        df = pd.DataFrame(all_records)
        df.to_parquet(MANIFEST_PARQUET, index=False)
        print(f"Saved canonical Parquet: {MANIFEST_PARQUET}")
    except Exception as e:
        csv_path = MANIFEST_PARQUET.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Parquet engine unavailable ({e}). Saved canonical CSV: {csv_path}")

    # Dataset distribution
    print("\nCorpus Distribution by Dataset:")
    print(df["dataset"].value_counts().to_string())

    print("\nCorpus Distribution by City:")
    print(df["city"].value_counts().to_string())

    print("=" * 70)


if __name__ == "__main__":
    main()
