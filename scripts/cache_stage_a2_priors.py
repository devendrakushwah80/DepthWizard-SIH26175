"""Build and numerically validate the deterministic frozen DAV2 cache for A2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.dav2_wrapper import DepthAnythingV2Wrapper


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
PREPROCESSING = {
    "model_id": MODEL_ID,
    "processor": "AutoImageProcessor.from_pretrained(model_id)",
    "processor_resize": "model-default deterministic resize/rescale/normalize",
    "requested_model_input": [518, 518],
    "output_interpolation": "torch bilinear align_corners=False",
    "output_dimensions": [1024, 1024],
    "cache_dtype": "float32",
}


def read_rows(paths: list[Path]) -> list[list[str]]:
    seen: set[str] = set()
    rows: list[list[str]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] not in seen:
                seen.add(parts[0])
                rows.append(parts)
    return rows


def read_rgb(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        array = np.asarray(handle[list(handle.keys())[0]])
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return array


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def valid_cache(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        return array.shape == (1024, 1024) and array.dtype == np.float32 and np.isfinite(array).all()
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[
            Path("data/gamus/splits/stage_a2/train_core.txt"),
            Path("data/gamus/splits/stage_a2/hpo_dev.txt"),
            Path("data/gamus/splits/stage_a2/final_dev_val.txt"),
        ],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/gamus/dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/gamus/cache_dav2"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/player1_stage_a2/dav2_cache"))
    parser.add_argument("--validation-samples", type=int, default=5)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifests)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = None
    metadata: list[dict] = []
    generated = 0
    started = time.time()
    for index, row in enumerate(rows, 1):
        sid, rgb_relpath = row[0], row[2]
        cache_path = args.cache_dir / f"{sid}_dav2.npy"
        status = "cached"
        if not valid_cache(cache_path):
            if wrapper is None:
                wrapper = DepthAnythingV2Wrapper(model_id=MODEL_ID, device=device)
            result = wrapper.predict_relative_depth(
                read_rgb(args.data_dir / rgb_relpath),
                target_size=(1024, 1024),
                input_size=(518, 518),
            )
            np.save(cache_path, result["raw_depth"].astype(np.float32))
            status = "generated"
            generated += 1
        array = np.load(cache_path, mmap_mode="r")
        metadata.append(
            {
                "sample_id": sid,
                "source_rgb_relpath": rgb_relpath,
                "cache_path": str(cache_path),
                "cache_status": status,
                "dtype": str(array.dtype),
                "height": int(array.shape[0]),
                "width": int(array.shape[1]),
                "cache_sha256": sha256(cache_path),
            }
        )
        if index % 25 == 0 or index == len(rows):
            print(f"DAV2 cache {index}/{len(rows)} generated={generated}", flush=True)

    # Recompute a deterministic spread of samples and compare to the cache.
    if wrapper is None:
        wrapper = DepthAnythingV2Wrapper(model_id=MODEL_ID, device=device)
    validation_indices = np.linspace(0, len(rows) - 1, min(args.validation_samples, len(rows)), dtype=int)
    validation = []
    for index in validation_indices:
        row = rows[int(index)]
        live = wrapper.predict_relative_depth(
            read_rgb(args.data_dir / row[2]),
            target_size=(1024, 1024),
            input_size=(518, 518),
        )["raw_depth"].astype(np.float32)
        cached = np.load(args.cache_dir / f"{row[0]}_dav2.npy").astype(np.float32)
        difference = np.abs(live - cached)
        validation.append(
            {
                "sample_id": row[0],
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
                "allclose_rtol_1e-5_atol_1e-6": bool(np.allclose(live, cached, rtol=1e-5, atol=1e-6)),
            }
        )

    with (args.out_dir / "cache_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    config = {
        **PREPROCESSING,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "model_parameters": wrapper.total_params,
        "sample_count": len(rows),
        "generated_count": generated,
        "preexisting_count": len(rows) - generated,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "cache_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out_dir / "cache_validation.json").write_text(
        json.dumps(
            {
                "samples": validation,
                "all_samples_pass": all(row["allclose_rtol_1e-5_atol_1e-6"] for row in validation),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not all(row["allclose_rtol_1e-5_atol_1e-6"] for row in validation):
        raise SystemExit("DAV2 cache numerical validation failed")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
