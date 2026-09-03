"""Download and verify every GAMUS file referenced by a tile manifest.

Unlike the original Stage A2 helper, this command never regenerates or shuffles
the manifest.  It is therefore safe to resume without silently changing the
experimental population.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download


def read_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            entries.extend((parts[0], relpath) for relpath in parts[2:5])
    return entries


def valid_h5(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
            return bool(keys) and handle[keys[0]].size > 0
    except (OSError, ValueError):
        return False


def download_one(relpath: str, data_dir: Path, retries: int) -> dict:
    destination = data_dir / relpath
    if valid_h5(destination):
        return {"relpath": relpath, "status": "cached", "error": None}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            hf_hub_download(
                repo_id="earthflow/GAMUS",
                filename=relpath,
                repo_type="dataset",
                local_dir=str(data_dir),
            )
            if not valid_h5(destination):
                raise OSError(f"downloaded H5 failed validation: {destination}")
            return {"relpath": relpath, "status": "downloaded", "error": None}
        except Exception as exc:  # external service errors vary by hub version
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return {"relpath": relpath, "status": "failed", "error": repr(last_error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-dir", default=Path("data/gamus/dataset"), type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--retries", default=3, type=int)
    args = parser.parse_args()

    requested = read_manifest(args.manifest)
    # A manifest may refer to the same file more than once; download it once.
    unique_paths = sorted({relpath for _, relpath in requested})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, relpath, args.data_dir, args.retries): relpath
            for relpath in unique_paths
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 100 == 0 or index == len(futures):
                counts = {
                    status: sum(r["status"] == status for r in results)
                    for status in ("cached", "downloaded", "failed")
                }
                print(f"{index}/{len(futures)} {counts}", flush=True)

    summary = {
        "manifest": str(args.manifest),
        "unique_files": len(unique_paths),
        "elapsed_seconds": time.time() - started,
        "cached": sum(r["status"] == "cached" for r in results),
        "downloaded": sum(r["status"] == "downloaded" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "failures": [r for r in results if r["status"] == "failed"],
    }
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
