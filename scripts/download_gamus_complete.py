"""
DepthWizard (SIH26175) — Autonomous Resumable GAMUS Corpus Downloader
Author: DepthWizard Phase 2 Pipeline
Target: earthflow/GAMUS (Hugging Face Datasets)

Downloads all missing publicly accessible DC and PHL training and validation records
with integrity validation (h5py verification, sha256 checksums, and storage safety guards).
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

import h5py
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "earthflow/GAMUS"
LOCAL_BASE = Path("data/gamus/dataset")
MANIFEST_OUT = Path("data/m3/manifests/gamus_download_manifest.jsonl")
STATUS_FILE = Path("outputs/m3/PHASE2_STATUS.json")
MIN_FREE_DISK_GB = 15.0


def check_free_disk_gb() -> float:
    _, _, free = shutil.disk_usage(".")
    return free / (1024 ** 3)


def is_valid_h5(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
            return bool(keys) and handle[keys[0]].size > 0
    except Exception:
        return False


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def update_status(downloaded_count: int, total_missing: int, bytes_downloaded: int):
    if not STATUS_FILE.exists():
        return
    try:
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)
        status["last_update"] = datetime.now(timezone.utc).isoformat()
        status["free_disk_gb"] = round(check_free_disk_gb(), 2)
        status["dataset_download_status"]["gamus"] = {
            "total_to_download": total_missing,
            "downloaded": downloaded_count,
            "bytes_downloaded_mb": round(bytes_downloaded / (1024 * 1024), 2),
            "state": "DOWNLOADING" if downloaded_count < total_missing else "COMPLETED"
        }
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to update status file: {e}")


def download_file(relpath: str, retries: int = 3) -> dict:
    dest = LOCAL_BASE / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if is_valid_h5(dest):
        return {"relpath": relpath, "status": "already_present", "size": dest.stat().st_size, "error": None}
    
    last_err = None
    for attempt in range(retries + 1):
        # Storage safety check
        if check_free_disk_gb() < MIN_FREE_DISK_GB:
            return {"relpath": relpath, "status": "aborted_low_disk", "size": 0, "error": "Disk space below safety threshold"}
            
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=relpath,
                repo_type="dataset",
                local_dir=str(LOCAL_BASE),
            )
            if is_valid_h5(dest):
                digest = compute_sha256(dest)
                return {
                    "relpath": relpath,
                    "status": "downloaded",
                    "size": dest.stat().st_size,
                    "sha256": digest,
                    "error": None
                }
            else:
                raise OSError(f"Corrupt H5 file downloaded: {dest}")
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                
    return {"relpath": relpath, "status": "failed", "size": 0, "error": str(last_err)}


def main():
    print("=" * 70)
    print("DepthWizard SIH26175 — Autonomous GAMUS Downloader")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    free_gb = check_free_disk_gb()
    print(f"Current free disk space: {free_gb:.2f} GB (Threshold: {MIN_FREE_DISK_GB} GB)")
    if free_gb < MIN_FREE_DISK_GB:
        print("[FATAL] Insufficient disk space to start download.")
        sys.exit(1)

    api = HfApi()
    print(f"Fetching authoritative remote file list from {REPO_ID}...")
    repo_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    print(f"Total files in remote repository: {len(repo_files)}")

    # Filter for non-NYC training & validation files
    target_files = []
    for f in repo_files:
        if not (f.endswith("_RGB.h5") or f.endswith("_AGL.h5") or f.endswith("_CLS.h5")):
            continue
        fname = f.split("/")[-1]
        city = fname.split("_")[0]
        # Only DC and PHL are training/dev candidates. NYC is sealed for external evaluation.
        if city in ("DC", "PHL"):
            target_files.append(f)

    print(f"Total legitimate training candidate files (DC + PHL): {len(target_files)}")

    # Check which files are missing or incomplete
    missing_files = []
    already_present = 0
    for f in target_files:
        dest = LOCAL_BASE / f
        if is_valid_h5(dest):
            already_present += 1
        else:
            missing_files.append(f)

    print(f"Already present and valid: {already_present}")
    print(f"Missing files to download: {len(missing_files)}")

    if not missing_files:
        print("[INFO] 100% of target training files are already present and verified!")
        update_status(0, 0, 0)
        return

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest_handle = open(MANIFEST_OUT, "a", encoding="utf-8")

    downloaded_count = 0
    bytes_downloaded = 0
    errors = []

    workers = 12
    print(f"Starting parallel download with {workers} worker threads...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(download_file, f): f for f in missing_files}
        
        for i, fut in enumerate(as_completed(future_map), 1):
            res = fut.result()
            if res["status"] == "downloaded":
                downloaded_count += 1
                bytes_downloaded += res["size"]
                manifest_handle.write(json.dumps(res) + "\n")
                manifest_handle.flush()
            elif res["status"] == "failed":
                errors.append(res)
                print(f"[ERROR] Failed {res['relpath']}: {res['error']}")
            elif res["status"] == "aborted_low_disk":
                print("[ALERT] Disk space threshold reached. Stopping download gracefully.")
                executor.shutdown(wait=False, cancel_futures=True)
                break

            if i % 50 == 0 or i == len(missing_files):
                elapsed = time.time() - start_time
                mb = bytes_downloaded / (1024 * 1024)
                rate = mb / elapsed if elapsed > 0 else 0
                pct = (i / len(missing_files)) * 100
                print(f"Progress: {i}/{len(missing_files)} ({pct:.1f}%) | "
                      f"Downloaded: {mb:.1f} MB ({rate:.2f} MB/s) | "
                      f"Errors: {len(errors)} | "
                      f"Free Disk: {check_free_disk_gb():.2f} GB")
                update_status(downloaded_count, len(missing_files), bytes_downloaded)

    manifest_handle.close()
    elapsed = time.time() - start_time
    total_mb = bytes_downloaded / (1024 * 1024)
    print("=" * 70)
    print("GAMUS DOWNLOAD SUMMARY:")
    print(f"Total downloaded files: {downloaded_count}")
    print(f"Total downloaded volume: {total_mb:.2f} MB ({total_mb / 1024:.2f} GB)")
    print(f"Total time elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"Average download throughput: {total_mb / elapsed if elapsed > 0 else 0:.2f} MB/s")
    print(f"Errors encountered: {len(errors)}")
    print(f"Final free disk space: {check_free_disk_gb():.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
