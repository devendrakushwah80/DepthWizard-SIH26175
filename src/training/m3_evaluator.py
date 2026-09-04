"""
DepthWizard (SIH26175) — M3 Comprehensive Evaluator
Author: DepthWizard Phase 2 Pipeline

Implements the official evaluation metrics:
- Overall: MAE, RMSE, R2, Pearson, Spearman, Bias, Median AE, NMAD, Within 1/2/5/10m
- Height Buckets: 0-2m, 2-10m, 10-20m, 20-50m, >=50m
- Domain & City breakdowns
- Multi-objective Pareto selection score
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from scipy import stats


def compute_metrics_array(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    valid = (mask > 0) & np.isfinite(target) & np.isfinite(pred) & (target >= 0.0)
    if valid.sum() == 0:
        return {"error": "No valid pixels"}

    p = np.clip(pred[valid], 0.0, None)
    t = target[valid]

    diff = p - t
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    med_ae = float(np.median(abs_diff))
    nmad = float(1.4826 * np.median(np.abs(diff - np.median(diff))))

    # R2
    t_var = np.var(t)
    if t_var > 1e-6:
        r2 = float(1.0 - (np.mean(diff ** 2) / t_var))
    else:
        r2 = 0.0

    # Pearson & Spearman correlations (subsample to 20k points if too large for speed)
    if len(p) > 20_000:
        sub_idx = np.random.choice(len(p), size=20_000, replace=False)
        p_sub, t_sub = p[sub_idx], t[sub_idx]
    else:
        p_sub, t_sub = p, t

    try:
        pearson, _ = stats.pearsonr(p_sub, t_sub)
        pearson = float(pearson) if not np.isnan(pearson) else 0.0
    except Exception:
        pearson = 0.0

    try:
        spearman, _ = stats.spearmanr(p_sub, t_sub)
        spearman = float(spearman) if not np.isnan(spearman) else 0.0
    except Exception:
        spearman = 0.0

    # Within thresholds
    within_1m = float(np.mean(abs_diff <= 1.0) * 100.0)
    within_2m = float(np.mean(abs_diff <= 2.0) * 100.0)
    within_5m = float(np.mean(abs_diff <= 5.0) * 100.0)
    within_10m = float(np.mean(abs_diff <= 10.0) * 100.0)

    # Height buckets
    buckets = {
        "0_2m": (0.0, 2.0),
        "2_10m": (2.0, 10.0),
        "10_20m": (10.0, 20.0),
        "20_50m": (20.0, 50.0),
        "ge_50m": (50.0, float("inf")),
    }

    bucket_stats = {}
    for b_name, (low, high) in buckets.items():
        b_idx = (t >= low) & (t < high)
        n_pix = int(b_idx.sum())
        if n_pix > 0:
            b_p = p[b_idx]
            b_t = t[b_idx]
            b_diff = b_p - b_t
            b_mae = float(np.mean(np.abs(b_diff)))
            b_rmse = float(np.sqrt(np.mean(b_diff ** 2)))
            b_bias = float(np.mean(b_diff))
            bucket_stats[b_name] = {
                "pixels": n_pix,
                "mae": b_mae,
                "rmse": b_rmse,
                "bias": b_bias,
                "gt_mean": float(np.mean(b_t)),
                "pred_mean": float(np.mean(b_p)),
            }
        else:
            bucket_stats[b_name] = {
                "pixels": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "bias": 0.0,
                "gt_mean": 0.0,
                "pred_mean": 0.0,
            }

    # Multi-objective Pareto selection score (lower is better)
    # Penalizes high MAE, poor R2, and underperformance on tall buildings (10-20, 20-50, >=50m)
    mae_10_20 = bucket_stats["10_20m"]["mae"]
    mae_20_50 = bucket_stats["20_50m"]["mae"]
    mae_ge_50 = bucket_stats["ge_50m"]["mae"] if bucket_stats["ge_50m"]["pixels"] > 0 else 0.0
    
    score = (
        mae
        + 0.5 * max(0.0, 1.0 - r2)
        + 0.25 * mae_10_20
        + 0.35 * mae_20_50
        + 0.15 * mae_ge_50
        + 0.20 * abs(bias)
    )

    return {
        "total_pixels": int(len(p)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "bias": bias,
        "median_ae": med_ae,
        "nmad": nmad,
        "within_1m": within_1m,
        "within_2m": within_2m,
        "within_5m": within_5m,
        "within_10m": within_10m,
        "height_buckets": bucket_stats,
        "selection_score": float(score),
    }


def evaluate_model_on_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    city_records: Dict[str, List[Dict[str, Any]]] = {}

    count = 0
    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device)
            prior = batch["prior"].to(device)
            target = batch["agl"].numpy()
            mask = batch["mask"].numpy()
            gsd_m = batch.get("gsd_m")
            gsd_known = batch.get("gsd_known")

            if gsd_m is not None:
                gsd_m = gsd_m.to(device)
            if gsd_known is not None:
                gsd_known = gsd_known.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if hasattr(model, "enable_gsd_conditioning") and model.enable_gsd_conditioning:
                    pred = model(rgb, prior, gsd_m=gsd_m, gsd_known=gsd_known)
                else:
                    pred = model(rgb, prior)

            pred_np = pred.cpu().float().numpy()

            B = rgb.shape[0]
            for b in range(B):
                p_b = pred_np[b, 0]
                t_b = target[b, 0]
                m_b = mask[b, 0]
                city = batch["city"][b]

                all_preds.append(p_b)
                all_targets.append(t_b)
                all_masks.append(m_b)

                if city not in city_records:
                    city_records[city] = {"preds": [], "targets": [], "masks": []}
                city_records[city]["preds"].append(p_b)
                city_records[city]["targets"].append(t_b)
                city_records[city]["masks"].append(m_b)

                count += 1
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break

    # Concatenate all valid pixels for global evaluation
    flat_pred = np.concatenate([p.ravel() for p in all_preds])
    flat_target = np.concatenate([t.ravel() for t in all_targets])
    flat_mask = np.concatenate([m.ravel() for m in all_masks])

    global_metrics = compute_metrics_array(flat_pred, flat_target, flat_mask)

    # City-level metrics
    city_metrics = {}
    for c, c_data in city_records.items():
        c_pred = np.concatenate([p.ravel() for p in c_data["preds"]])
        c_target = np.concatenate([t.ravel() for t in c_data["targets"]])
        c_mask = np.concatenate([m.ravel() for m in c_data["masks"]])
        city_metrics[c] = compute_metrics_array(c_pred, c_target, c_mask)

    return {
        "global": global_metrics,
        "by_city": city_metrics,
        "sample_count": count,
    }
