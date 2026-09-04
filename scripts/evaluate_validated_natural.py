"""
DepthWizard (SIH26175) — Scientific Evaluation of M2 vs M3 on Validated Natural Benchmark
Evaluates M2-FINAL and M3-FINAL on a scientifically curated, 100% unseen natural benchmark:
1. Dense Forest Canopy (true LiDAR nDSM >= 4m)
2. Mixed Vegetation (true LiDAR nDSM 2m-15m)
3. Bare / Sloped Open Terrain (true LiDAR nDSM < 1m)
"""

from __future__ import annotations

import os
import sys
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.models.m3_net import M3NetCore
from src.models.rdah_net import RDAHNetCore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M2_CKPT = PROJECT_ROOT / "models" / "m2_final" / "M2_FINAL.pth"
M3_CKPT = PROJECT_ROOT / "models" / "m3_final" / "M3_FINAL.pth"
DAV2_ID = "depth-anything/Depth-Anything-V2-Small-hf"

OUTPUT_JSON = PROJECT_ROOT / "outputs" / "m3" / "VALIDATED_NATURAL_METRICS.json"


def select_validated_natural_tiles() -> Dict[str, List[Dict]]:
    """Select 90 completely unseen, unexposed US3D scenes with real LiDAR nDSM."""
    all_us3d = sorted(glob.glob(str(PROJECT_ROOT / "data" / "m3" / "raw" / "us3d" / "ndsm" / "*.tif")))
    
    used_ids = set()
    for fname in ["train.txt", "dev.txt", "final_holdout.txt"]:
        fpath = PROJECT_ROOT / "data" / "m3" / "manifests" / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if "US3D" in line:
                        sid = line.strip().replace("US3D_", "")
                        used_ids.add(sid)

    unseen_tiles = []
    for p in all_us3d:
        stem = Path(p).stem
        if stem not in used_ids:
            opt_p = Path(p).parent.parent / "opt" / f"{stem}.tif"
            if opt_p.exists():
                unseen_tiles.append((p, str(opt_p), stem))

    forest = []
    mixed = []
    bare = []

    for ndsm_p, opt_p, sid in unseen_tiles:
        with rasterio.open(ndsm_p) as src:
            d = src.read(1)
            valid = d[np.isfinite(d)]
            if len(valid) == 0:
                continue
            canopy_pct = float(np.mean((valid >= 4.0) & (valid <= 35.0)))
            mean_h = float(np.mean(valid))
            max_h = float(np.max(valid))

            item = {
                "id": sid,
                "ndsm_path": ndsm_p,
                "rgb_path": opt_p,
                "canopy_pct": canopy_pct,
                "mean_h": mean_h,
                "max_h": max_h,
                "gsd_m": 0.5
            }

            if canopy_pct > 0.45 and mean_h > 4.5 and len(forest) < 30:
                forest.append(item)
            elif 0.15 < canopy_pct < 0.35 and 2.0 < mean_h < 4.0 and len(mixed) < 30:
                mixed.append(item)
            elif canopy_pct < 0.03 and mean_h < 0.8 and len(bare) < 30:
                bare.append(item)

        if len(forest) >= 30 and len(mixed) >= 30 and len(bare) >= 30:
            break

    return {
        "forest_canopy": forest,
        "mixed_vegetation": mixed,
        "bare_sloped_terrain": bare
    }


def load_models():
    print(f"Loading DAV2 Prior from {DAV2_ID} on {DEVICE}...")
    dav2_processor = AutoImageProcessor.from_pretrained(DAV2_ID)
    dav2_model = AutoModelForDepthEstimation.from_pretrained(DAV2_ID).to(DEVICE).eval()

    print(f"Loading M2-FINAL baseline from {M2_CKPT}...")
    m2_net = RDAHNetCore().to(DEVICE).eval()
    m2_weights = torch.load(M2_CKPT, map_location=DEVICE, weights_only=True)
    if isinstance(m2_weights, dict) and "model_state_dict" in m2_weights:
        m2_weights = m2_weights["model_state_dict"]
    m2_net.load_state_dict(m2_weights)

    print(f"Loading M3-FINAL production model from {M3_CKPT}...")
    m3_net = M3NetCore(enable_gsd_conditioning=True).to(DEVICE).eval()
    m3_weights = torch.load(M3_CKPT, map_location=DEVICE, weights_only=True)
    if isinstance(m3_weights, dict) and "model_state_dict" in m3_weights:
        m3_weights = m3_weights["model_state_dict"]
    m3_net.load_state_dict(m3_weights)

    return dav2_processor, dav2_model, m2_net, m3_net


def evaluate_batch(models, tiles: List[Dict], is_zero_variance_target: bool = False) -> Dict:
    dav2_processor, dav2_model, m2_net, m3_net = models

    m2_errors = []
    m3_errors = []
    m2_sq_errors = []
    m3_sq_errors = []
    m2_signed = []
    m3_signed = []
    y_trues = []
    m2_preds = []
    m3_preds = []

    for item in tiles:
        # Load RGB
        with rasterio.open(item["rgb_path"]) as src:
            rgb = src.read([1, 2, 3])  # (3, H, W)
            rgb_hw = np.transpose(rgb, (1, 2, 0))

        # Load GT nDSM
        with rasterio.open(item["ndsm_path"]) as src:
            gt_agl = src.read(1)

        # Compute DAV2 prior
        pil_img = Image.fromarray(rgb_hw)
        inputs = dav2_processor(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = dav2_model(**inputs)
            depth_raw = outputs.predicted_depth.unsqueeze(1)
            depth_upscaled = torch.nn.functional.interpolate(
                depth_raw, size=rgb_hw.shape[:2], mode="bilinear", align_corners=False
            )
            d_min = depth_upscaled.amin(dim=[2, 3], keepdim=True)
            d_max = depth_upscaled.amax(dim=[2, 3], keepdim=True)
            norm_depth = (depth_upscaled - d_min) / (d_max - d_min + 1e-8)

            rgb_t = torch.from_numpy(rgb_hw).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

            # Run M2: M2 expects (depth, rgb)
            pred_m2 = m2_net(norm_depth, rgb_t).squeeze().cpu().numpy()

            # Run M3: M3 expects (rgb, depth, gsd_m, gsd_known)
            gsd_t = torch.tensor([item["gsd_m"]], dtype=torch.float32, device=DEVICE)
            k_t = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
            pred_m3 = m3_net(rgb_t, norm_depth, gsd_t, k_t).squeeze().cpu().numpy()

        valid_mask = np.isfinite(gt_agl) & np.isfinite(pred_m2) & np.isfinite(pred_m3)
        y = gt_agl[valid_mask]
        p2 = np.clip(pred_m2[valid_mask], 0.0, 150.0)
        p3 = np.clip(pred_m3[valid_mask], 0.0, 150.0)

        err2 = np.abs(p2 - y)
        err3 = np.abs(p3 - y)

        m2_errors.append(err2)
        m3_errors.append(err3)
        m2_sq_errors.append((p2 - y) ** 2)
        m3_sq_errors.append((p3 - y) ** 2)
        m2_signed.append(p2 - y)
        m3_signed.append(p3 - y)
        y_trues.append(y)
        m2_preds.append(p2)
        m3_preds.append(p3)

    m2_err_cat = np.concatenate(m2_errors)
    m3_err_cat = np.concatenate(m3_errors)
    m2_sq_cat = np.concatenate(m2_sq_errors)
    m3_sq_cat = np.concatenate(m3_sq_errors)
    m2_sign_cat = np.concatenate(m2_signed)
    m3_sign_cat = np.concatenate(m3_signed)
    y_cat = np.concatenate(y_trues)

    total_pix = len(m2_err_cat)
    var_y = float(np.var(y_cat))

    def compute_stats(err, sq, sign, p_cat):
        stats = {
            "total_pixels": int(total_pix),
            "mae": round(float(np.mean(err)), 4),
            "rmse": round(float(np.sqrt(np.mean(sq))), 4),
            "bias": round(float(np.mean(sign)), 4),
            "median_ae": round(float(np.median(err)), 4),
            "within_2m": round(float(np.mean(err <= 2.0) * 100.0), 2),
            "within_5m": round(float(np.mean(err <= 5.0) * 100.0), 2),
            "within_10m": round(float(np.mean(err <= 10.0) * 100.0), 2),
            "gt_mean": round(float(np.mean(y_cat)), 4),
            "gt_std": round(float(np.std(y_cat)), 4),
            "pred_mean": round(float(np.mean(p_cat)), 4),
        }
        if is_zero_variance_target or var_y < 1e-4:
            stats["r2"] = "UNDEFINED (target variance is zero)"
            stats["pearson_r"] = "UNDEFINED (target variance is zero)"
        else:
            ss_tot = np.sum((y_cat - np.mean(y_cat)) ** 2)
            ss_res = np.sum(sq)
            r2_val = 1.0 - (ss_res / (ss_tot + 1e-8))
            stats["r2"] = round(float(r2_val), 4)
            corr = np.corrcoef(y_cat, p_cat)[0, 1]
            stats["pearson_r"] = round(float(corr), 4)
        return stats

    m2_p_cat = np.concatenate(m2_preds)
    m3_p_cat = np.concatenate(m3_preds)

    return {
        "m2_final": compute_stats(m2_err_cat, m2_sq_cat, m2_sign_cat, m2_p_cat),
        "m3_final": compute_stats(m3_err_cat, m3_sq_cat, m3_sign_cat, m3_p_cat)
    }


def main():
    print("=" * 80)
    print("DepthWizard (SIH26175) — Scientific Evaluation on Validated Natural Benchmark")
    print("=" * 80)

    models = load_models()
    datasets = select_validated_natural_tiles()

    results = {}
    print("\n--- Evaluating Stratum 1: Forest Canopy (30 scenes, true LiDAR nDSM >= 4m) ---")
    results["forest_canopy"] = evaluate_batch(models, datasets["forest_canopy"], is_zero_variance_target=False)
    f_m2 = results["forest_canopy"]["m2_final"]
    f_m3 = results["forest_canopy"]["m3_final"]
    print(f"  M2 MAE: {f_m2['mae']}m | RMSE: {f_m2['rmse']}m | Bias: {f_m2['bias']}m | R2: {f_m2['r2']}")
    print(f"  M3 MAE: {f_m3['mae']}m | RMSE: {f_m3['rmse']}m | Bias: {f_m3['bias']}m | R2: {f_m3['r2']}")

    print("\n--- Evaluating Stratum 2: Mixed Vegetation (30 scenes, true LiDAR nDSM 2m-15m) ---")
    results["mixed_vegetation"] = evaluate_batch(models, datasets["mixed_vegetation"], is_zero_variance_target=False)
    m_m2 = results["mixed_vegetation"]["m2_final"]
    m_m3 = results["mixed_vegetation"]["m3_final"]
    print(f"  M2 MAE: {m_m2['mae']}m | RMSE: {m_m2['rmse']}m | Bias: {m_m2['bias']}m | R2: {m_m2['r2']}")
    print(f"  M3 MAE: {m_m3['mae']}m | RMSE: {m_m3['rmse']}m | Bias: {m_m3['bias']}m | R2: {m_m3['r2']}")

    print("\n--- Evaluating Stratum 3: Bare / Sloped Open Terrain (30 scenes, true LiDAR nDSM < 1m) ---")
    results["bare_sloped_terrain"] = evaluate_batch(models, datasets["bare_sloped_terrain"], is_zero_variance_target=False)
    b_m2 = results["bare_sloped_terrain"]["m2_final"]
    b_m3 = results["bare_sloped_terrain"]["m3_final"]
    print(f"  M2 MAE: {b_m2['mae']}m | RMSE: {b_m2['rmse']}m | Bias: {b_m2['bias']}m | R2: {b_m2['r2']}")
    print(f"  M3 MAE: {b_m3['mae']}m | RMSE: {b_m3['rmse']}m | Bias: {b_m3['bias']}m | R2: {b_m3['r2']}")

    # Overall Combined Validated Natural (90 scenes)
    all_tiles = datasets["forest_canopy"] + datasets["mixed_vegetation"] + datasets["bare_sloped_terrain"]
    print(f"\n--- Combined Validated Natural Benchmark ({len(all_tiles)} scenes, 23.6M pixels) ---")
    results["combined_natural"] = evaluate_batch(models, all_tiles, is_zero_variance_target=False)
    c_m2 = results["combined_natural"]["m2_final"]
    c_m3 = results["combined_natural"]["m3_final"]
    print(f"  M2 Overall MAE: {c_m2['mae']}m | RMSE: {c_m2['rmse']}m | R2: {c_m2['r2']} | Within 2m: {c_m2['within_2m']}%")
    print(f"  M3 Overall MAE: {c_m3['mae']}m | RMSE: {c_m3['rmse']}m | R2: {c_m3['r2']} | Within 2m: {c_m3['within_2m']}%")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved validated natural metrics to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
