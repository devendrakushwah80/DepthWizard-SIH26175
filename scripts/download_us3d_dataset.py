"""
DepthWizard (SIH26175) — Autonomous US3D / DFC2019 Downloader & Auditor
Author: DepthWizard Phase 2 Pipeline
Target: YixuanMa/US3D1w (Optical RGB) and YixuanMa/nDSM (Ground-truth nDSM)

Acquires matched satellite RGB GeoTIFFs and LiDAR-derived metric AGL/nDSM GeoTIFFs
for Jacksonville (JAX) and Omaha (OMA) with integrity validation.
"""

from __future__ import annotations

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download, HfApi

OPT_REPO = "YixuanMa/US3D1w"
NDSM_REPO = "YixuanMa/nDSM"
LOCAL_OPT_DIR = Path("data/m3/raw/us3d/opt")
LOCAL_NDSM_DIR = Path("data/m3/raw/us3d/ndsm")
MANIFEST_OUT = Path("data/m3/manifests/us3d_manifest.jsonl")
MIN_FREE_DISK_GB = 15.0


def check_free_disk_gb() -> float:
    _, _, free = shutil.disk_usage(".")
    return free / (1024 ** 3)


def is_valid_tif(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    # Check minimum file header
    try:
        with open(path, "rb") as f:
            header = f.read(4)
            return header in (b"II*\x00", b"MM\x00*")  # Standard TIFF magic bytes
    except Exception:
        return False


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_pair(pair_info: dict) -> dict:
    opt_rel = pair_info["opt_rel"]
    ndsm_rel = pair_info["ndsm_rel"]
    tile_id = pair_info["tile_id"]

    opt_dest = LOCAL_OPT_DIR / f"{tile_id}.tif"
    ndsm_dest = LOCAL_NDSM_DIR / f"{tile_id}.tif"

    if is_valid_tif(opt_dest) and is_valid_tif(ndsm_dest):
        return {
            "tile_id": tile_id,
            "status": "already_present",
            "opt_path": str(opt_dest),
            "ndsm_path": str(ndsm_dest),
            "error": None
        }

    if check_free_disk_gb() < MIN_FREE_DISK_GB:
        return {"tile_id": tile_id, "status": "aborted_low_disk", "error": "Disk space safety limit reached"}

    try:
        # Download optical
        if not is_valid_tif(opt_dest):
            hf_hub_download(
                repo_id=OPT_REPO,
                filename=opt_rel,
                repo_type="dataset",
                local_dir=str(LOCAL_OPT_DIR.parent),
            )
            # Find and move to canonical destination
            downloaded = LOCAL_OPT_DIR.parent / opt_rel
            if downloaded.exists():
                shutil.move(str(downloaded), str(opt_dest))

        # Download nDSM
        if not is_valid_tif(ndsm_dest):
            hf_hub_download(
                repo_id=NDSM_REPO,
                filename=ndsm_rel,
                repo_type="dataset",
                local_dir=str(LOCAL_NDSM_DIR.parent),
            )
            downloaded = LOCAL_NDSM_DIR.parent / ndsm_rel
            if downloaded.exists():
                shutil.move(str(downloaded), str(ndsm_dest))

        if is_valid_tif(opt_dest) and is_valid_tif(ndsm_dest):
            return {
                "tile_id": tile_id,
                "city": pair_info["city"],
                "status": "downloaded",
                "opt_path": str(opt_dest),
                "ndsm_path": str(ndsm_dest),
                "opt_size": opt_dest.stat().st_size,
                "ndsm_size": ndsm_dest.stat().st_size,
                "opt_sha256": compute_sha256(opt_dest),
                "ndsm_sha256": compute_sha256(ndsm_dest),
                "error": None
            }
        else:
            raise OSError("Downloaded TIFF failed validation")
    except Exception as exc:
        return {"tile_id": tile_id, "status": "failed", "error": str(exc)}


def main(target_count_per_city: int = 1000):
    print("=" * 70)
    print("DepthWizard SIH26175 — US3D / DFC2019 Dataset Downloader")
    print("=" * 70)

    LOCAL_OPT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_NDSM_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    print("Querying remote file trees for US3D optical and nDSM repositories...")
    ndsm_files = api.list_repo_files(NDSM_REPO, repo_type="dataset")
    opt_files = api.list_repo_files(OPT_REPO, repo_type="dataset")

    # Map tile_id to optical relative path
    opt_map = {}
    for f in opt_files:
        if f.endswith(".tif"):
            tile_id = Path(f).stem
            opt_map[tile_id] = f

    # Match with nDSM
    pairs = []
    for f in ndsm_files:
        if f.startswith("gt_nDSM/") and f.endswith(".tif"):
            tile_id = Path(f).stem
            if tile_id in opt_map:
                city = tile_id.split("_")[0]
                pairs.append({
                    "tile_id": tile_id,
                    "city": city,
                    "opt_rel": opt_map[tile_id],
                    "ndsm_rel": f
                })

    print(f"Total matched remote pairs found: {len(pairs)}")
    jax_pairs = [p for p in pairs if p["city"] == "JAX"]
    oma_pairs = [p for p in pairs if p["city"] == "OMA"]
    print(f"  Jacksonville (JAX): {len(jax_pairs)}")
    print(f"  Omaha (OMA): {len(oma_pairs)}")

    # Select target subset (balanced across cities)
    selected = jax_pairs[:target_count_per_city] + oma_pairs[:target_count_per_city]
    print(f"Target download quota: {len(selected)} pairs ({target_count_per_city} JAX + {target_count_per_city} OMA)")

    downloaded = 0
    errors = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(download_pair, p): p for p in selected}
        with open(MANIFEST_OUT, "a", encoding="utf-8") as manifest_handle:
            for i, fut in enumerate(as_completed(future_map), 1):
                res = fut.result()
                if res["status"] in ("downloaded", "already_present"):
                    downloaded += 1
                    manifest_handle.write(json.dumps(res) + "\n")
                    manifest_handle.flush()
                else:
                    errors.append(res)
                    print(f"[WARN] Failed {res.get('tile_id')}: {res.get('error')}")

                if i % 100 == 0 or i == len(selected):
                    elapsed = time.time() - start_time
                    rate = downloaded / elapsed if elapsed > 0 else 0
                    print(f"US3D Progress: {i}/{len(selected)} ({i/len(selected)*100:.1f}%) | "
                          f"Valid pairs: {downloaded} ({rate:.1f} pairs/s) | "
                          f"Free Disk: {check_free_disk_gb():.2f} GB")

    print("=" * 70)
    print(f"US3D Download Complete: {downloaded} pairs verified.")
    print("=" * 70)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(target_count_per_city=count)
