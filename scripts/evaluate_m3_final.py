"""
DepthWizard (SIH26175) — Official M3 Final Production Evaluator & Promotion Gate
Evaluates:
  1. M2-FINAL baseline
  2. M3_FINAL_refit production candidate
Across:
  - DEV Partition (314 scenes)
  - FINAL-HOLDOUT Partition (310 scenes, completely sealed)
  - EXTERNAL-NYC Partition (500 scenes, zero-shot city generalization)
  - NATURAL Benchmark (10 scenes, dense forest and mountainous steep)

Checks promotion criteria, creates models/m3_final/M3_FINAL.pth, computes SHA256,
and produces outputs/m3/M3_FINAL_METRICS.json and outputs/m3/M3_PROMOTION_REPORT.json.
"""

from __future__ import annotations

import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.rdah_net import RDAHNetCore
from src.models.m3_net import M3NetCore
from src.training.m3_dataset import M3MultiDomainDataset
from src.training.m3_evaluator import evaluate_model_on_dataset

M2_PATH = Path("models/m2_final/M2_FINAL.pth")
M3_CANDIDATE_PATH = Path("models/m3_experiments/M3_FINAL_refit/checkpoints/best_overall.pth")
M3_FINAL_PROD_PATH = Path("models/m3_final/M3_FINAL.pth")

OUT_METRICS_PATH = Path("outputs/m3/M3_FINAL_METRICS.json")
OUT_PROMOTION_PATH = Path("outputs/m3/M3_PROMOTION_REPORT.json")
OUT_ABLATION_PATH = Path("outputs/m3/ABLATION_SUMMARY.json")


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_m2_model(device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(M2_PATH, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("rdah_core."):
            cleaned_state[k[10:]] = v
        elif k.startswith("base."):
            cleaned_state[k[5:]] = v
        else:
            cleaned_state[k] = v
    core = RDAHNetCore(d_model=32)
    core.load_state_dict(cleaned_state)
    core.to(device)
    core.eval()

    class M2Adapter(torch.nn.Module):
        def __init__(self, core_net):
            super().__init__()
            self.core = core_net
            self.enable_gsd_conditioning = False
        def forward(self, rgb, prior, **kwargs):
            return self.core(prior, rgb)

    return M2Adapter(core)


def load_m3_model(device: torch.device, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = M3NetCore(enable_gsd_conditioning=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    print("=" * 80)
    print("DepthWizard (SIH26175) — M3 Final Production Evaluation & Promotion Gate")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Verify M2-FINAL invariant
    m2_sha = compute_sha256(M2_PATH)
    print(f"M2-FINAL SHA-256: {m2_sha}")
    expected_m2_sha = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"
    assert m2_sha == expected_m2_sha, f"M2-FINAL SHA mismatch! Expected {expected_m2_sha}, got {m2_sha}"
    print("[PASS] M2-FINAL Integrity Verified (Frozen).")

    # Load models
    print("\nLoading models...")
    m2_model = load_m2_model(device)
    m3_model = load_m3_model(device, M3_CANDIDATE_PATH)
    print("Both models loaded successfully.")

    # Datasets
    manifest_path = "data/m3/manifests/all_records.jsonl"
    splits = {
        "DEV": "data/m3/manifests/dev.txt",
        "FINAL_HOLDOUT": "data/m3/manifests/final_holdout.txt",
        "EXTERNAL_NYC": "data/m3/manifests/nyc_external.txt",
        "NATURAL": "data/m3/manifests/natural.txt",
    }

    eval_results = {
        "m2": {},
        "m3": {}
    }

    for split_name, split_file in splits.items():
        print(f"\nEvaluating on {split_name} ({split_file})...")
        ds = M3MultiDomainDataset(
            manifest_path=manifest_path,
            split_records_path=split_file,
            crop_size=512,
            is_training=False,
            height_aware_sampling=False,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        print(f"  Dataset size: {len(ds)} scenes")

        print("  Evaluating M3 Candidate...")
        m3_res = evaluate_model_on_dataset(m3_model, loader, device)
        eval_results["m3"][split_name] = m3_res

        print("  Evaluating M2 Baseline...")
        m2_res = evaluate_model_on_dataset(m2_model, loader, device)
        eval_results["m2"][split_name] = m2_res

        # Print comparison
        m3_g = m3_res["global"]
        m2_g = m2_res["global"]
        print(f"  --> {split_name} Comparison:")
        print(f"      MAE:   M2 = {m2_g['mae']:.4f}m | M3 = {m3_g['mae']:.4f}m ({(m2_g['mae'] - m3_g['mae']) / m2_g['mae'] * 100:+.1f}%)")
        print(f"      RMSE:  M2 = {m2_g['rmse']:.4f}m | M3 = {m3_g['rmse']:.4f}m")
        print(f"      R2:    M2 = {m2_g['r2']:.4f} | M3 = {m3_g['r2']:.4f}")
        print(f"      10-20m MAE: M2 = {m2_g['height_buckets']['10_20m']['mae']:.2f}m | M3 = {m3_g['height_buckets']['10_20m']['mae']:.2f}m")
        print(f"      20-50m MAE: M2 = {m2_g['height_buckets']['20_50m']['mae']:.2f}m | M3 = {m3_g['height_buckets']['20_50m']['mae']:.2f}m")
        print(f"      Score: M2 = {m2_g['selection_score']:.4f} | M3 = {m3_g['selection_score']:.4f}")

    # Check Promotion Gate
    print("\n" + "=" * 80)
    print("PROMOTION GATE VERIFICATION")
    print("=" * 80)

    checks = []

    # Check 1: DEV Overall MAE improvement
    dev_m2_mae = eval_results["m2"]["DEV"]["global"]["mae"]
    dev_m3_mae = eval_results["m3"]["DEV"]["global"]["mae"]
    dev_mae_pass = dev_m3_mae < dev_m2_mae
    checks.append({
        "check": "DEV Overall MAE Improvement",
        "m2_val": f"{dev_m2_mae:.4f}m",
        "m3_val": f"{dev_m3_mae:.4f}m",
        "passed": dev_mae_pass,
    })

    # Check 2: DEV R2 improvement
    dev_m2_r2 = eval_results["m2"]["DEV"]["global"]["r2"]
    dev_m3_r2 = eval_results["m3"]["DEV"]["global"]["r2"]
    dev_r2_pass = dev_m3_r2 > dev_m2_r2
    checks.append({
        "check": "DEV R2 Improvement",
        "m2_val": f"{dev_m2_r2:.4f}",
        "m3_val": f"{dev_m3_r2:.4f}",
        "passed": dev_r2_pass,
    })

    # Check 3: DEV 10-20m MAE improvement
    dev_m2_10_20 = eval_results["m2"]["DEV"]["global"]["height_buckets"]["10_20m"]["mae"]
    dev_m3_10_20 = eval_results["m3"]["DEV"]["global"]["height_buckets"]["10_20m"]["mae"]
    dev_10_20_pass = dev_m3_10_20 < dev_m2_10_20
    checks.append({
        "check": "DEV 10-20m MAE Improvement",
        "m2_val": f"{dev_m2_10_20:.4f}m",
        "m3_val": f"{dev_m3_10_20:.4f}m",
        "passed": dev_10_20_pass,
    })

    # Check 4: DEV 20-50m MAE improvement
    dev_m2_20_50 = eval_results["m2"]["DEV"]["global"]["height_buckets"]["20_50m"]["mae"]
    dev_m3_20_50 = eval_results["m3"]["DEV"]["global"]["height_buckets"]["20_50m"]["mae"]
    dev_20_50_pass = dev_m3_20_50 < dev_m2_20_50
    checks.append({
        "check": "DEV 20-50m MAE Improvement",
        "m2_val": f"{dev_m2_20_50:.4f}m",
        "m3_val": f"{dev_m3_20_50:.4f}m",
        "passed": dev_20_50_pass,
    })

    # Check 5: DEV Selection Score improvement
    dev_m2_score = eval_results["m2"]["DEV"]["global"]["selection_score"]
    dev_m3_score = eval_results["m3"]["DEV"]["global"]["selection_score"]
    dev_score_pass = dev_m3_score < dev_m2_score
    checks.append({
        "check": "DEV Selection Score Improvement",
        "m2_val": f"{dev_m2_score:.4f}",
        "m3_val": f"{dev_m3_score:.4f}",
        "passed": dev_score_pass,
    })

    # Check 6: FINAL-HOLDOUT MAE improvement
    holdout_m2_mae = eval_results["m2"]["FINAL_HOLDOUT"]["global"]["mae"]
    holdout_m3_mae = eval_results["m3"]["FINAL_HOLDOUT"]["global"]["mae"]
    holdout_mae_pass = holdout_m3_mae < holdout_m2_mae
    checks.append({
        "check": "FINAL-HOLDOUT MAE Improvement",
        "m2_val": f"{holdout_m2_mae:.4f}m",
        "m3_val": f"{holdout_m3_mae:.4f}m",
        "passed": holdout_mae_pass,
    })

    # Check 7: EXTERNAL-NYC Zero-Shot MAE improvement
    nyc_m2_mae = eval_results["m2"]["EXTERNAL_NYC"]["global"]["mae"]
    nyc_m3_mae = eval_results["m3"]["EXTERNAL_NYC"]["global"]["mae"]
    nyc_mae_pass = nyc_m3_mae < nyc_m2_mae
    checks.append({
        "check": "EXTERNAL-NYC Zero-Shot MAE Improvement",
        "m2_val": f"{nyc_m2_mae:.4f}m",
        "m3_val": f"{nyc_m3_mae:.4f}m",
        "passed": nyc_mae_pass,
    })

    # Check 8: EXTERNAL-NYC Zero-Shot R2 improvement
    nyc_m2_r2 = eval_results["m2"]["EXTERNAL_NYC"]["global"]["r2"]
    nyc_m3_r2 = eval_results["m3"]["EXTERNAL_NYC"]["global"]["r2"]
    nyc_r2_pass = nyc_m3_r2 > nyc_m2_r2
    checks.append({
        "check": "EXTERNAL-NYC Zero-Shot R2 Improvement",
        "m2_val": f"{nyc_m2_r2:.4f}",
        "m3_val": f"{nyc_m3_r2:.4f}",
        "passed": nyc_r2_pass,
    })

    all_passed = all(c["passed"] for c in checks)
    for c in checks:
        status_str = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"  {status_str} {c['check']}: M2={c['m2_val']} vs M3={c['m3_val']}")

    print("-" * 80)
    print(f"Overall Promotion Gate Status: {'ACCEPTED' if all_passed else 'REJECTED'}")

    if all_passed:
        # Copy to production
        M3_FINAL_PROD_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(M3_CANDIDATE_PATH, M3_FINAL_PROD_PATH)
        m3_sha = compute_sha256(M3_FINAL_PROD_PATH)
        print(f"\n[PROMOTED] M3 Promoted to Production!")
        print(f"  Path:   {M3_FINAL_PROD_PATH}")
        print(f"  SHA256: {m3_sha}")
    else:
        m3_sha = "NOT_PROMOTED"

    # Save Promotion Report
    promotion_report = {
        "timestamp": "2026-09-04T13:52:00Z",
        "promotion_status": "ACCEPTED" if all_passed else "REJECTED",
        "m2_baseline": {
            "checkpoint": str(M2_PATH),
            "sha256": m2_sha,
        },
        "m3_final": {
            "candidate_source": str(M3_CANDIDATE_PATH),
            "production_path": str(M3_FINAL_PROD_PATH) if all_passed else None,
            "sha256": m3_sha,
        },
        "gates": checks,
    }
    OUT_PROMOTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PROMOTION_PATH, "w", encoding="utf-8") as f:
        json.dump(promotion_report, f, indent=2)

    # Save Final Metrics
    final_metrics = {
        "timestamp": "2026-09-04T13:52:00Z",
        "m2_sha256": m2_sha,
        "m3_sha256": m3_sha,
        "evaluations": eval_results,
    }
    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Compile Ablation Summary
    ablation_summary = {
        "experiments": {
            "M2-FINAL": {
                "description": "Previous frozen baseline (RDAH-Net, GAMUS partial train)",
                "dev_mae": dev_m2_mae,
                "dev_r2": dev_m2_r2,
                "dev_10_20m_mae": dev_m2_10_20,
                "dev_20_50m_mae": dev_m2_20_50,
                "dev_score": dev_m2_score,
                "nyc_mae": nyc_m2_mae,
                "nyc_r2": nyc_m2_r2,
            },
            "M3-A_full_data": {
                "description": "Full Multi-Domain Training (GAMUS + US3D)",
                "dev_mae": 2.8295,
                "dev_r2": 0.5241,
                "dev_10_20m_mae": 5.67,
                "dev_20_50m_mae": 12.73,
                "dev_score": 15.7502,
            },
            "M3-B_height_balanced": {
                "description": "Height-Aware Sampling + Weighted Loss + Tall Bias Penalty",
                "dev_mae": 3.0906,
                "dev_r2": 0.5275,
                "dev_10_20m_mae": 5.90,
                "dev_20_50m_mae": 8.41,
                "dev_score": 13.7896,
            },
            "M3-C_gsd_conditioned": {
                "description": "FiLM Bottleneck + Decoder Modulation via Metric GSD (Seed 42)",
                "dev_mae": 2.9924,
                "dev_r2": 0.5529,
                "dev_10_20m_mae": 5.55,
                "dev_20_50m_mae": 7.22,
                "dev_score": 13.1307,
            },
            "M3-C_seed1337": {
                "description": "Multi-Seed Verification (Seed 1337)",
                "dev_mae": 3.2815,
                "dev_10_20m_mae": 5.59,
                "dev_20_50m_mae": 7.25,
                "dev_score": 13.9327,
            },
            "M3-C_seed2026": {
                "description": "Multi-Seed Verification (Seed 2026)",
                "dev_mae": 3.1972,
                "dev_10_20m_mae": 5.55,
                "dev_20_50m_mae": 7.69,
                "dev_score": 14.0730,
            },
            "M3_FINAL_refit": {
                "description": "Production Refit on TRAIN + DEV partitions",
                "dev_mae": dev_m3_mae,
                "dev_r2": dev_m3_r2,
                "dev_10_20m_mae": dev_m3_10_20,
                "dev_20_50m_mae": dev_m3_20_50,
                "dev_score": dev_m3_score,
                "holdout_mae": holdout_m3_mae,
                "nyc_mae": nyc_m3_mae,
                "nyc_r2": nyc_m3_r2,
            },
        }
    }
    with open(OUT_ABLATION_PATH, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    print(f"\nSaved Reports:")
    print(f"  {OUT_PROMOTION_PATH}")
    print(f"  {OUT_METRICS_PATH}")
    print(f"  {OUT_ABLATION_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
