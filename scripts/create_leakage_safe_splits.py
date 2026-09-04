"""
DepthWizard (SIH26175) — Leakage-Safe Multi-Domain Dataset Partitioner
Author: DepthWizard Phase 2 Pipeline

Creates strictly disjoint, leakage-free geographic splits:
- TRAIN: ~80% of geographic super-blocks across PHL, DC, JAX, OMA
- DEV: ~10% of geographic super-blocks (for architecture/loss selection & HPO)
- FINAL-HOLDOUT: ~10% of geographic super-blocks (sealed until final evaluation)
- EXTERNAL-NYC: 500 samples sealed unseen zero-shot generalization test
- NATURAL-BENCHMARK: 10 AOIs (forest, mountain, hilly)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd

MANIFEST_JSONL = Path("data/m3/manifests/all_records.jsonl")
SPLITS_DIR = Path("data/m3/manifests")
AUDIT_DIR = Path("outputs/m3/data_audit")
SEED = 42


def main():
    print("=" * 70)
    print("DepthWizard SIH26175 — Creating Leakage-Safe Multi-Domain Splits")
    print("=" * 70)

    if not MANIFEST_JSONL.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_JSONL}. Run build_unified_m3_manifest.py first.")

    records = []
    with open(MANIFEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} total records from manifest.")

    # Separate special splits
    nyc_records = [r for r in records if r["city"] == "NYC"]
    natural_records = [r for r in records if r.get("split") == "NATURAL_BENCHMARK"]
    candidates = [r for r in records if r["city"] != "NYC" and r.get("split") != "NATURAL_BENCHMARK"]

    print(f"External NYC records: {len(nyc_records)}")
    print(f"Natural domain benchmark records: {len(natural_records)}")
    print(f"Legitimate candidate records for TRAIN/DEV/HOLDOUT: {len(candidates)}")

    # Group candidates by city and geographic super-block
    city_groups = defaultdict(lambda: defaultdict(list))
    for r in candidates:
        city = r["city"]
        group = r["geographic_group"]
        city_groups[city][group].append(r)

    train_records = []
    dev_records = []
    holdout_records = []

    rng = random.Random(SEED)

    overlap_groups = {}
    group_to_split = {}

    for city, groups_dict in sorted(city_groups.items()):
        groups = sorted(groups_dict.keys())
        rng.shuffle(groups)

        n_groups = len(groups)
        # Allocate: ~80% train, ~10% dev, ~10% holdout (at least 1 dev, 1 holdout)
        n_dev = max(1, int(round(n_groups * 0.10)))
        n_holdout = max(1, int(round(n_groups * 0.10)))
        n_train = n_groups - n_dev - n_holdout

        dev_g = set(groups[:n_dev])
        holdout_g = set(groups[n_dev:n_dev + n_holdout])
        train_g = set(groups[n_dev + n_holdout:])

        print(f"City: {city:<5} | Super-blocks: {n_groups:<4} | Train: {len(train_g):<3} | Dev: {len(dev_g):<3} | Holdout: {len(holdout_g):<3}")

        for g in train_g:
            train_records.extend(groups_dict[g])
            group_to_split[g] = "TRAIN"
            overlap_groups[g] = [r["record_id"] for r in groups_dict[g]]

        for g in dev_g:
            dev_records.extend(groups_dict[g])
            group_to_split[g] = "DEV"
            overlap_groups[g] = [r["record_id"] for r in groups_dict[g]]

        for g in holdout_g:
            holdout_records.extend(groups_dict[g])
            group_to_split[g] = "FINAL_HOLDOUT"
            overlap_groups[g] = [r["record_id"] for r in groups_dict[g]]

    print("-" * 70)
    print(f"SPLIT POPULATION:")
    print(f"  TRAIN:         {len(train_records):<6} records (100% of legitimate training partition)")
    print(f"  DEV:           {len(dev_records):<6} records (independent validation for HPO/loss)")
    print(f"  FINAL-HOLDOUT: {len(holdout_records):<6} records (sealed until final evaluation)")
    print(f"  EXTERNAL-NYC:  {len(nyc_records):<6} records (unseen zero-shot test)")
    print(f"  NATURAL-BENCH: {len(natural_records):<6} records (forest/mountain benchmark)")

    # ZERO LEAKAGE AUDIT
    train_g_set = {r["geographic_group"] for r in train_records}
    dev_g_set = {r["geographic_group"] for r in dev_records}
    holdout_g_set = {r["geographic_group"] for r in holdout_records}

    leakage_train_dev = train_g_set.intersection(dev_g_set)
    leakage_train_holdout = train_g_set.intersection(holdout_g_set)
    leakage_dev_holdout = dev_g_set.intersection(holdout_g_set)

    total_leakage = len(leakage_train_dev) + len(leakage_train_holdout) + len(leakage_dev_holdout)

    print("-" * 70)
    print("GEOGRAPHIC LEAKAGE AUDIT:")
    print(f"  TRAIN / DEV overlap:         {len(leakage_train_dev)}")
    print(f"  TRAIN / HOLDOUT overlap:     {len(leakage_train_holdout)}")
    print(f"  DEV / HOLDOUT overlap:       {len(leakage_dev_holdout)}")
    print(f"  TOTAL KNOWN LEAKAGE:         {total_leakage}")

    if total_leakage != 0:
        raise RuntimeError(f"CRITICAL: Geographic leakage detected ({total_leakage} overlapping groups)!")
    print("  STATUS: 100% LEAKAGE-FREE GEOGRAPHIC ISOLATION VERIFIED.")

    # Write split files
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_DIR / "train.txt", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(r["record_id"] + "\n")

    with open(SPLITS_DIR / "dev.txt", "w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(r["record_id"] + "\n")

    with open(SPLITS_DIR / "final_holdout.txt", "w", encoding="utf-8") as f:
        for r in holdout_records:
            f.write(r["record_id"] + "\n")

    with open(SPLITS_DIR / "nyc_external.txt", "w", encoding="utf-8") as f:
        for r in nyc_records:
            f.write(r["record_id"] + "\n")

    # Write audit reports
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    leakage_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "seed": SEED,
        "total_records": len(records),
        "train_count": len(train_records),
        "dev_count": len(dev_records),
        "final_holdout_count": len(holdout_records),
        "nyc_external_count": len(nyc_records),
        "natural_benchmark_count": len(natural_records),
        "known_leakage": 0,
        "leakage_train_dev": list(leakage_train_dev),
        "leakage_train_holdout": list(leakage_train_holdout),
        "leakage_dev_holdout": list(leakage_dev_holdout),
        "status": "PASS_ZERO_LEAKAGE"
    }
    with open(AUDIT_DIR / "leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)

    with open(AUDIT_DIR / "overlap_groups.json", "w", encoding="utf-8") as f:
        json.dump(overlap_groups, f, indent=2)

    print(f"\nSaved split manifests to: {SPLITS_DIR}")
    print(f"Saved audit reports to: {AUDIT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
