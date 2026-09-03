"""Full-resolution 1024px Stage A2 evaluation with 512/256 Hanning blending."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch

from src.models.rdah_net import RDAHNetCore
from src.training.stage_a2_engine import BUCKETS, RegressionAccumulator, _tile_metrics


DATA_DIR = Path("data/gamus/dataset")
CACHE_DIR = Path("data/gamus/cache_dav2")


def _read_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[list(handle.keys())[0]])


def _manifest_rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # On Windows a read-only watcher can briefly hold the destination without
    # delete sharing. Retry the atomic replace instead of failing evaluation.
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def _load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["model"] if isinstance(payload, dict) and "model" in payload else payload


def hanning_window(size: int = 512) -> np.ndarray:
    # A small positive floor retains Hanning weighting without undefined
    # zero-weight corner pixels at the outer image boundary.
    return np.clip(np.outer(np.hanning(size), np.hanning(size)), 0.05, 1.0).astype(np.float32)


def sliding_window_prediction(
    model: torch.nn.Module,
    rgb: np.ndarray,
    depth: np.ndarray,
    device: str,
) -> np.ndarray:
    if rgb.shape[:2] != (1024, 1024) or depth.shape != (1024, 1024):
        raise ValueError(f"full-tile inference requires 1024x1024 inputs, got {rgb.shape}, {depth.shape}")
    window = hanning_window()
    canvas = np.zeros((1024, 1024), dtype=np.float32)
    weights = np.zeros_like(canvas)
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
    with torch.inference_mode():
        for top in (0, 256, 512):
            for left in (0, 256, 512):
                rgb_patch = rgb_tensor[:, :, top : top + 512, left : left + 512].to(device)
                depth_patch = depth_tensor[:, :, top : top + 512, left : left + 512].to(device)
                with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                    prediction = model(depth_patch, rgb_patch)
                prediction = prediction.clamp_min(0.0)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"non-finite full-tile patch at top={top}, left={left}")
                patch = prediction.squeeze().float().cpu().numpy()
                canvas[top : top + 512, left : left + 512] += patch * window
                weights[top : top + 512, left : left + 512] += window
    result = canvas / np.maximum(weights, 1e-6)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite reconstructed full-tile prediction")
    return result


def evaluate_full_tile_checkpoint(
    *,
    model_name: str,
    checkpoint: Path,
    output_parameterization: str,
    manifest: Path,
    out_dir: Path,
    progress_callback: Callable[[dict], None] | None = None,
    model_index: int = 1,
    model_total: int = 1,
) -> dict:
    """Evaluate one RDAH checkpoint identically over reconstructed full tiles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions_float32"
    prediction_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RDAHNetCore(
        d_model=32,
        num_heads=4,
        output_parameterization=output_parameterization,
    )
    model.load_state_dict(_load_state(checkpoint))
    model.to(device).eval()

    rows = _manifest_rows(manifest)
    global_acc = RegressionAccumulator(sample_cap=500_000, seed=701)
    building_acc = RegressionAccumulator(sample_cap=350_000, seed=702)
    bucket_acc = [RegressionAccumulator(sample_cap=200_000, seed=710 + index) for index in range(5)]
    bucket_tile_count = [0] * len(BUCKETS)
    per_tile: list[dict] = []
    skipped_tiles: list[str] = []
    started = time.time()
    for tile_index, row in enumerate(rows, 1):
        sample_id = row[0]
        rgb = _read_h5(DATA_DIR / row[2])
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0
        gt = np.squeeze(_read_h5(DATA_DIR / row[3])).astype(np.float32)
        semantic = np.squeeze(_read_h5(DATA_DIR / row[4])).astype(np.int64)
        valid = np.isfinite(gt) & (gt >= 0.0) & (semantic >= 0) & (semantic <= 6)
        if not np.any(valid):
            skipped_tiles.append(sample_id)
            continue
        depth_path = CACHE_DIR / f"{sample_id}_dav2.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"missing frozen DAV2 cache: {depth_path}")
        prediction_path = prediction_dir / f"{sample_id}.npy"
        if prediction_path.exists():
            prediction = np.load(prediction_path).astype(np.float32, copy=False)
            if prediction.shape != (1024, 1024) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid resumable prediction cache: {prediction_path}")
        else:
            depth = np.load(depth_path, mmap_mode="r").astype(np.float32)
            prediction = sliding_window_prediction(model, rgb, depth, device)
            np.save(prediction_path, prediction.astype(np.float32))

        pred_valid = prediction[valid]
        gt_valid = gt[valid]
        semantic_valid = semantic[valid]
        global_acc.update(pred_valid, gt_valid)
        building = semantic_valid == 3
        building_acc.update(pred_valid[building], gt_valid[building])
        for bucket_index, (_, low, high) in enumerate(BUCKETS):
            selected = (gt_valid >= low) & (gt_valid < high)
            if np.any(selected):
                bucket_tile_count[bucket_index] += 1
            bucket_acc[bucket_index].update(pred_valid[selected], gt_valid[selected])
        per_tile.append({"sample_id": sample_id, **_tile_metrics(pred_valid, gt_valid)})

        running_mae = global_acc.sum_abs_error / max(1, global_acc.count)
        progress = {
            "model_name": model_name,
            "model_index": model_index,
            "model_total": model_total,
            "tile_number": tile_index,
            "tile_total": len(rows),
            "valid_tiles": len(per_tile),
            "skipped_tiles": len(skipped_tiles),
            "running_mae_m": running_mae,
            "elapsed_seconds": time.time() - started,
        }
        _dump(out_dir / "evaluation_progress.json", progress)
        if progress_callback is not None:
            progress_callback(progress)

    metrics = global_acc.metrics()
    metrics["dataset"] = {
        "manifest": str(manifest),
        "manifest_tiles": len(rows),
        "valid_tile_count": len(per_tile),
        "skipped_tile_count": len(skipped_tiles),
        "skipped_tiles": skipped_tiles,
        "valid_pixel_count": metrics["pixel_count"],
        "inference_window": [512, 512],
        "step": 256,
        "grid": [3, 3],
        "blend": "Hanning weighted with 0.05 boundary floor",
    }
    metrics["building"] = building_acc.metrics()
    total_pixels = max(1, metrics["pixel_count"])
    metrics["height_buckets"] = {}
    for bucket_index, ((bucket_name, _, _), accumulator) in enumerate(zip(BUCKETS, bucket_acc)):
        bucket_metrics = accumulator.metrics()
        bucket_metrics["support_percentage"] = bucket_metrics.get("pixel_count", 0) / total_pixels * 100.0
        bucket_metrics["tile_count_with_support"] = bucket_tile_count[bucket_index]
        metrics["height_buckets"][bucket_name] = bucket_metrics
    tile_mae = np.asarray([row["mae_m"] for row in per_tile], dtype=np.float64)
    metrics["tile"] = {
        "tile_count": len(per_tile),
        "mean_tile_mae_m": float(tile_mae.mean()),
        "median_tile_mae_m": float(np.median(tile_mae)),
        "std_tile_mae_m": float(tile_mae.std()),
        "iqr_tile_mae_m": float(np.percentile(tile_mae, 75) - np.percentile(tile_mae, 25)),
        "p90_tile_mae_m": float(np.percentile(tile_mae, 90)),
        "p95_tile_mae_m": float(np.percentile(tile_mae, 95)),
    }
    metrics["model"] = {
        "name": model_name,
        "checkpoint": str(checkpoint),
        "output_parameterization": output_parameterization,
    }
    metrics["runtime_seconds"] = time.time() - started
    _dump(out_dir / "metrics.json", metrics)
    _dump(out_dir / "per_tile_metrics.json", per_tile)
    with (out_dir / "per_tile_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_tile[0]))
        writer.writeheader()
        writer.writerows(per_tile)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metrics, "per_tile": per_tile}
