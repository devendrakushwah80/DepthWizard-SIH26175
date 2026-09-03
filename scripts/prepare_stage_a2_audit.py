"""Create Stage A2 splits, integrity/leakage evidence, and target audit plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CLASS_NAMES = {
    0: "Others",
    1: "Ground",
    2: "Low vegetation",
    3: "Building",
    4: "Water",
    5: "Road",
    6: "Tree",
}
BUCKETS = (
    ("0-2m", 0.0, 2.0),
    ("2-10m", 2.0, 10.0),
    ("10-20m", 10.0, 20.0),
    ("20-50m", 20.0, 50.0),
    (">=50m", 50.0, np.inf),
)
GRID = (0, 256, 512)


def sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def read_rows(path: Path) -> list[list[str]]:
    return [line.strip().split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def read_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError("empty H5")
        return np.asarray(handle[keys[0]])


def inspect_triplet(row: list[str], data_dir: Path, read_targets: bool = True) -> dict:
    result = {"sample_id": row[0], "corrupt": False, "errors": []}
    expected = ((row[2], "rgb"), (row[3], "height"), (row[4], "semantic"))
    arrays: dict[str, np.ndarray] = {}
    for relpath, kind in expected:
        path = data_dir / relpath
        if not path.is_file():
            result["corrupt"] = True
            result["errors"].append(f"missing:{relpath}")
            continue
        try:
            with h5py.File(path, "r") as handle:
                keys = list(handle.keys())
                if not keys or handle[keys[0]].size == 0:
                    raise ValueError("empty dataset")
                result[f"{kind}_shape"] = list(handle[keys[0]].shape)
            if read_targets or kind == "rgb":
                arrays[kind] = read_array(path)
        except Exception as exc:
            result["corrupt"] = True
            result["errors"].append(f"{kind}:{type(exc).__name__}:{exc}")
    result["arrays"] = arrays
    return result


def bucket_indices(height: np.ndarray) -> np.ndarray:
    result = np.zeros(height.shape, dtype=np.uint8)
    result[height >= 2.0] = 1
    result[height >= 10.0] = 2
    result[height >= 20.0] = 3
    result[height >= 50.0] = 4
    return result


def approximate_median_from_centimetre_hist(hist: np.ndarray) -> float | None:
    total = int(hist.sum())
    if total == 0:
        return None
    index = int(np.searchsorted(np.cumsum(hist), (total - 1) // 2 + 1))
    return index / 100.0


def stratified_hpo_split(tile_stats: list[dict], seed: int, per_city: int = 100) -> set[str]:
    chosen: set[str] = set()
    for city in ("PHL", "DC"):
        city_rows = [row for row in tile_stats if row["city"] == city]
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in city_rows:
            groups[row["tile_max_bucket"]].append(row)
        selected: list[dict] = []
        for rows in groups.values():
            rows.sort(key=lambda x: stable_key(x["sample_id"], seed))
            count = max(1, round(len(rows) * per_city / len(city_rows)))
            selected.extend(rows[:count])
        selected.sort(key=lambda x: stable_key(x["sample_id"], seed + 1))
        if len(selected) > per_city:
            selected = selected[:per_city]
        elif len(selected) < per_city:
            selected_ids = {row["sample_id"] for row in selected}
            remainder = [row for row in city_rows if row["sample_id"] not in selected_ids]
            remainder.sort(key=lambda x: stable_key(x["sample_id"], seed + 2))
            selected.extend(remainder[: per_city - len(selected)])
        chosen.update(row["sample_id"] for row in selected)
    return chosen


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/gamus/splits/stage_a2_train.txt"))
    parser.add_argument("--final-val", type=Path, default=Path("data/gamus/splits/stage_a1_val.txt"))
    parser.add_argument("--nyc", type=Path, default=Path("data/gamus/splits/nyc_unseen_500_test.txt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/gamus/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/player1_stage_a2"))
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/gamus/splits/stage_a2"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remote-phl", type=int, default=4398)
    parser.add_argument("--remote-dc", type=int, default=2159)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    analysis_dir = args.out_dir / "data_analysis"
    manifests_dir = args.manifests_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    pool_rows, val_rows, nyc_rows = map(read_rows, (args.pool, args.final_val, args.nyc))
    corrupt: list[dict] = []
    nodata: list[str] = []
    tile_stats: list[dict] = []
    crop_rows: list[dict] = []
    semantic_counts = np.zeros(7, dtype=np.int64)
    semantic_bucket = np.zeros((7, 5), dtype=np.int64)
    bucket_count = np.zeros(5, dtype=np.int64)
    bucket_sum = np.zeros(5, dtype=np.float64)
    bucket_max = np.full(5, -np.inf, dtype=np.float64)
    bucket_tiles = np.zeros(5, dtype=np.int64)
    bucket_building = np.zeros(5, dtype=np.int64)
    bucket_useful_crops = np.zeros(5, dtype=np.int64)
    # Quantised at 1 cm: medians are reproducible to +/- 0.005 m.
    bucket_hists = [np.zeros(1, dtype=np.int64) for _ in range(5)]
    height_plot_hist = np.zeros(401, dtype=np.int64)  # 0..100 m at 0.25 m, overflow last
    building_plot_hist = np.zeros(401, dtype=np.int64)

    for index, row in enumerate(pool_rows, 1):
        inspected = inspect_triplet(row, args.data_dir, read_targets=True)
        if inspected["corrupt"]:
            corrupt.append({k: v for k, v in inspected.items() if k != "arrays"})
            continue
        rgb = inspected["arrays"]["rgb"]
        height = np.squeeze(inspected["arrays"]["height"]).astype(np.float32)
        semantic = np.squeeze(inspected["arrays"]["semantic"]).astype(np.int64)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb_shape = (rgb.shape[1], rgb.shape[2], rgb.shape[0])
        else:
            rgb_shape = rgb.shape
        if height.shape != semantic.shape or tuple(rgb_shape[:2]) != height.shape:
            corrupt.append({"sample_id": row[0], "errors": ["shape_mismatch"], "corrupt": True})
            continue
        valid = np.isfinite(height) & (height >= 0.0) & (semantic >= 0) & (semantic <= 6)
        if not np.any(valid):
            nodata.append(row[0])
            continue
        values = height[valid]
        sem_values = semantic[valid]
        bidx = bucket_indices(values)
        max_height = float(values.max())
        tile_max_bucket = BUCKETS[int(bucket_indices(np.asarray([max_height]))[0])][0]
        tile_stats.append(
            {
                "sample_id": row[0],
                "city": row[0].split("_")[0],
                "valid_pixels": int(valid.sum()),
                "invalid_pixels": int(valid.size - valid.sum()),
                "max_height_m": max_height,
                "mean_height_m": float(values.mean()),
                "building_fraction": float(np.mean((semantic == 3) & valid)),
                "tile_max_bucket": tile_max_bucket,
            }
        )
        semantic_counts += np.bincount(sem_values, minlength=7)[:7]
        semantic_bucket += np.bincount(sem_values * 5 + bidx, minlength=35).reshape(7, 5)
        quantised = np.rint(np.clip(values, 0, 10000) * 100).astype(np.int64)
        for bucket_id in range(5):
            selected = bidx == bucket_id
            count = int(selected.sum())
            if count:
                bucket_count[bucket_id] += count
                bucket_sum[bucket_id] += float(values[selected].sum(dtype=np.float64))
                bucket_max[bucket_id] = max(bucket_max[bucket_id], float(values[selected].max()))
                bucket_tiles[bucket_id] += 1
                bucket_building[bucket_id] += int(np.sum(sem_values[selected] == 3))
                local_hist = np.bincount(quantised[selected])
                if local_hist.size > bucket_hists[bucket_id].size:
                    bucket_hists[bucket_id] = np.pad(
                        bucket_hists[bucket_id], (0, local_hist.size - bucket_hists[bucket_id].size)
                    )
                bucket_hists[bucket_id][: local_hist.size] += local_hist
        height_plot_hist += np.bincount(
            np.minimum((values / 0.25).astype(np.int64), 400), minlength=401
        )[:401]
        building_values = values[sem_values == 3]
        if building_values.size:
            building_plot_hist += np.bincount(
                np.minimum((building_values / 0.25).astype(np.int64), 400), minlength=401
            )[:401]

        for top in GRID:
            for left in GRID:
                h_crop = height[top : top + 512, left : left + 512]
                s_crop = semantic[top : top + 512, left : left + 512]
                v_crop = np.isfinite(h_crop) & (h_crop >= 0.0) & (s_crop >= 0) & (s_crop <= 6)
                crop_values = h_crop[v_crop]
                crop_bucket = bucket_indices(crop_values) if crop_values.size else np.empty(0, dtype=np.uint8)
                bucket_crop_counts = np.bincount(crop_bucket, minlength=5)[:5]
                bucket_useful_crops += bucket_crop_counts >= 64
                crop_rows.append(
                    {
                        "sample_id": row[0],
                        "top": top,
                        "left": left,
                        "valid_pixels": int(v_crop.sum()),
                        "max_gt": float(crop_values.max()) if crop_values.size else 0.0,
                        "building_fraction": float(np.mean((s_crop == 3) & v_crop)),
                        "random": 1,
                        "building_rich": int(np.mean((s_crop == 3) & v_crop) >= 0.05),
                        "gt10": int(np.sum(v_crop & (h_crop >= 10.0)) >= 64),
                        "gt20": int(np.sum(v_crop & (h_crop >= 20.0)) >= 32),
                        "rare_tall": int(np.sum(v_crop & (h_crop >= 50.0)) >= 8),
                    }
                )
        if index % 100 == 0 or index == len(pool_rows):
            print(f"audited {index}/{len(pool_rows)} pool tiles", flush=True)

    if corrupt or nodata:
        bad_ids = {row["sample_id"] for row in corrupt} | set(nodata)
        pool_rows = [row for row in pool_rows if row[0] not in bad_ids]
        tile_stats = [row for row in tile_stats if row["sample_id"] not in bad_ids]
        crop_rows = [row for row in crop_rows if row["sample_id"] not in bad_ids]

    hpo_ids = stratified_hpo_split(tile_stats, args.seed, per_city=100)
    train_rows = [row for row in pool_rows if row[0] not in hpo_ids]
    hpo_rows = [row for row in pool_rows if row[0] in hpo_ids]
    write_rows(manifests_dir / "train_core.txt", train_rows)
    write_rows(manifests_dir / "hpo_dev.txt", hpo_rows)
    write_rows(manifests_dir / "final_dev_val.txt", val_rows)
    write_rows(manifests_dir / "nyc_locked_test.txt", nyc_rows)

    with (analysis_dir / "tile_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_stats[0]))
        writer.writeheader()
        writer.writerows(tile_stats)
    with (analysis_dir / "crop_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crop_rows[0]))
        writer.writeheader()
        writer.writerows(crop_rows)

    bucket_records: list[dict] = []
    total_pixels = int(bucket_count.sum())
    for index, (name, _, _) in enumerate(BUCKETS):
        bucket_records.append(
            {
                "height_bucket": name,
                "pixel_count": int(bucket_count[index]),
                "pixel_pct": float(bucket_count[index] / total_pixels * 100) if total_pixels else 0.0,
                "tile_count": int(bucket_tiles[index]),
                "mean_height_m": float(bucket_sum[index] / bucket_count[index]) if bucket_count[index] else None,
                "median_height_m": approximate_median_from_centimetre_hist(bucket_hists[index]),
                "max_height_m": float(bucket_max[index]) if bucket_count[index] else None,
                "building_pixel_count": int(bucket_building[index]),
                "useful_512_crops": int(bucket_useful_crops[index]),
            }
        )
    with (analysis_dir / "height_bucket_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bucket_records[0]))
        writer.writeheader()
        writer.writerows(bucket_records)

    with (analysis_dir / "height_semantic_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["semantic_class", *[name for name, _, _ in BUCKETS], "total"])
        for class_id, class_name in CLASS_NAMES.items():
            writer.writerow([class_name, *semantic_bucket[class_id].tolist(), int(semantic_bucket[class_id].sum())])

    centers = np.arange(401) * 0.25
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(centers[:-1], height_plot_hist[:-1], linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("AGL height (m)")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Stage A2 training-pool height distribution (0-100 m)")
    fig.tight_layout()
    fig.savefig(analysis_dir / "height_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.bar([row["height_bucket"] for row in bucket_records], [row["pixel_pct"] for row in bucket_records])
    ax.set_ylabel("Valid pixels (%)")
    ax.set_title("Height bucket distribution")
    fig.tight_layout()
    fig.savefig(analysis_dir / "height_bucket_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(centers[:-1], building_plot_hist[:-1], linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Building-pixel AGL height (m)")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Building height distribution (0-100 m)")
    fig.tight_layout()
    fig.savefig(analysis_dir / "building_height_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.hist([row["max_height_m"] for row in tile_stats], bins=50)
    ax.set_xlabel("Tile maximum AGL height (m)")
    ax.set_ylabel("Tile count")
    ax.set_title("Tile maximum-height distribution")
    fig.tight_layout()
    fig.savefig(analysis_dir / "tile_max_height_histogram.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar([CLASS_NAMES[i] for i in range(7)], semantic_counts / semantic_counts.sum() * 100)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Valid pixels (%)")
    ax.set_title("Semantic distribution")
    fig.tight_layout()
    fig.savefig(analysis_dir / "semantic_distribution.png")
    plt.close(fig)

    sets = {
        "train_core": {row[0] for row in train_rows},
        "hpo_dev": {row[0] for row in hpo_rows},
        "final_dev_val": {row[0] for row in val_rows},
        "nyc": {row[0] for row in nyc_rows},
    }
    intersections = {
        f"{a}_intersect_{b}": sorted(sets[a] & sets[b])
        for a, b in (
            ("train_core", "hpo_dev"),
            ("train_core", "final_dev_val"),
            ("train_core", "nyc"),
            ("hpo_dev", "final_dev_val"),
            ("hpo_dev", "nyc"),
            ("final_dev_val", "nyc"),
        )
    }

    # RGB hashes provide a practical exact-content duplicate audit without reading
    # NYC labels.  They also detect duplicate imagery under different identifiers.
    rgb_hash_rows: list[dict] = []
    for split_name, rows in (
        ("train_core", train_rows),
        ("hpo_dev", hpo_rows),
        ("final_dev_val", val_rows),
        ("nyc", nyc_rows),
    ):
        for index, row in enumerate(rows, 1):
            rgb_hash_rows.append(
                {
                    "split": split_name,
                    "sample_id": row[0],
                    "rgb_relpath": row[2],
                    "rgb_sha256": sha256(args.data_dir / row[2]),
                }
            )
        print(f"hashed RGB content for {split_name}", flush=True)
    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rgb_hash_rows:
        hash_groups[row["rgb_sha256"]].append(row)
    cross_split_hash_duplicates = [
        rows for rows in hash_groups.values() if len({row["split"] for row in rows}) > 1
    ]
    with (analysis_dir / "tile_rgb_hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rgb_hash_rows[0]))
        writer.writeheader()
        writer.writerows(rgb_hash_rows)

    manifest_hashes = {
        name: sha256(manifests_dir / filename)
        for name, filename in {
            "train_core": "train_core.txt",
            "hpo_dev": "hpo_dev.txt",
            "final_dev_val": "final_dev_val.txt",
            "nyc_locked_test": "nyc_locked_test.txt",
        }.items()
    }
    audit = {
        "remote_inventory": {
            "phl_total_triplets": args.remote_phl,
            "dc_total_triplets": args.remote_dc,
            "phl_dc_total_triplets": args.remote_phl + args.remote_dc,
            "phl_train_capable_excluding_final_dev_val": args.remote_phl - 100,
            "dc_train_capable_excluding_final_dev_val": args.remote_dc - 100,
            "phl_dc_train_capable_excluding_final_dev_val": args.remote_phl + args.remote_dc - 200,
        },
        "fixed_a2_pool": len(pool_rows),
        "train_core": len(train_rows),
        "hpo_dev": len(hpo_rows),
        "final_dev_val": len(val_rows),
        "nyc_locked_manifest": len(nyc_rows),
        "city_counts": {
            name: dict(Counter(row[0].split("_")[0] for row in rows))
            for name, rows in (
                ("train_core", train_rows),
                ("hpo_dev", hpo_rows),
                ("final_dev_val", val_rows),
                ("nyc", nyc_rows),
            )
        },
        "duplicate_entries": {
            name: len(rows) - len({row[0] for row in rows})
            for name, rows in (
                ("pool", pool_rows),
                ("final_dev_val", val_rows),
                ("nyc", nyc_rows),
            )
        },
        "corrupted_samples": corrupt,
        "nodata_training_samples": nodata,
        "intersections": intersections,
        "cross_split_rgb_hash_duplicates": cross_split_hash_duplicates,
        "nyc_leakage_zero": not any(
            intersections[name]
            for name in (
                "train_core_intersect_nyc",
                "hpo_dev_intersect_nyc",
                "final_dev_val_intersect_nyc",
            )
        )
        and not any(
            any(row["split"] == "nyc" for row in group)
            and any(row["split"] != "nyc" for row in group)
            for group in cross_split_hash_duplicates
        ),
        "manifest_sha256": manifest_hashes,
        "height_bucket_statistics": bucket_records,
        "median_method": "1 cm quantised streaming histogram (maximum +/- 0.005 m quantisation error)",
        "useful_crop_definition": "fixed 512px crop contains at least 64 valid pixels in the bucket",
        "crop_pool_thresholds": {
            "building_rich": "building fraction >= 5%",
            "gt10": ">=64 valid pixels with GT >=10m",
            "gt20": ">=32 valid pixels with GT >=20m",
            "rare_tall": ">=8 valid pixels with GT >=50m",
        },
    }
    (args.out_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    experiment = {
        "experiment_id": f"player1_stage_a2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit() or "unavailable_not_a_git_checkout",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": [args.seed, 1337],
        "aborted_run_archive": None,
    }
    (args.out_dir / "experiment_metadata.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    report_lines = [
        "# Player 1 Stage A2 Data Audit",
        "",
        f"Audit timestamp (UTC): {experiment['timestamp_utc']}",
        "",
        "## Inventory",
        "",
        f"- Official repository PHL triplets: {args.remote_phl}",
        f"- Official repository DC triplets: {args.remote_dc}",
        f"- Train-capable after reserving FINAL-DEV-VAL: PHL {args.remote_phl - 100}; DC {args.remote_dc - 100}",
        f"- Fixed, downloaded A2 pool: {len(pool_rows)} (PHL 1,000; DC 1,000)",
        f"- TRAIN-CORE: {len(train_rows)}",
        f"- HPO-DEV: {len(hpo_rows)}",
        f"- FINAL-DEV-VAL: {len(val_rows)} (the untouched Stage A1 official validation manifest)",
        f"- NYC locked manifest: {len(nyc_rows)} rows; evaluable count remains sealed until the final one-shot run",
        "",
        "The 2,000-tile pool was selected before this clean run and was not regenerated. It doubles the M1 training population while retaining disk for deterministic DAV2 caches, HPO checkpoints, and full-resolution evaluation on the available 114 GB free workspace.",
        "",
        "## Integrity and duplicates",
        "",
        f"- Corrupted pool samples: {len(corrupt)}",
        f"- Nodata pool samples: {len(nodata)}",
        f"- Duplicate manifest entries: {sum(audit['duplicate_entries'].values())}",
        f"- Cross-split duplicate RGB hashes: {len(cross_split_hash_duplicates)}",
        "",
        "## Leakage",
        "",
        *[f"- {name}: {len(values)}" for name, values in intersections.items()],
        f"- NYC LEAKAGE = {'ZERO' if audit['nyc_leakage_zero'] else 'DETECTED'}",
        "",
        "Exact IDs, paths, SHA-256 hashes, target statistics, and crop-pool definitions are in the generated Stage A2 audit outputs.",
    ]
    (docs / "PLAYER1_STAGE_A2_DATA_AUDIT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in {"height_bucket_statistics", "cross_split_rgb_hash_duplicates", "corrupted_samples"}}, indent=2))


if __name__ == "__main__":
    main()
