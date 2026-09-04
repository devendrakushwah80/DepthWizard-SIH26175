"""
DepthWizard (SIH26175) — Official M2 Apples-to-Apples Baseline Evaluator
Evaluates frozen M2-FINAL (SHA256: 6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f)
on the exact same multi-domain DEV and NYC splits using the M3 unified evaluator.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.models.rdah_net import RDAHNetCore
from src.training.m3_dataset import M3MultiDomainDataset
from src.training.m3_evaluator import evaluate_model_on_dataset

M2_CHECKPOINT_PATH = Path("models/m2_final/M2_FINAL.pth")
OUT_FILE = Path("outputs/m3/m2_dev_baseline.json")


def main():
    print("=" * 70)
    print("DepthWizard SIH26175 — Evaluating Frozen M2 Baseline")
    print(f"M2 Checkpoint: {M2_CHECKPOINT_PATH}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load M2 checkpoint
    ckpt = torch.load(M2_CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    # Strip any prefix if needed (e.g. rdah_core.)
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("rdah_core."):
            cleaned_state[k[10:]] = v
        elif k.startswith("base."):
            cleaned_state[k[5:]] = v
        else:
            cleaned_state[k] = v

    model = RDAHNetCore(d_model=32)
    model.load_state_dict(cleaned_state)
    model.to(device)
    model.eval()
    print("M2-FINAL model loaded successfully into evaluator.")

    # Create wrapper to match forward signature (rgb, prior) -> (depth, img)
    class M2Adapter(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
            self.enable_gsd_conditioning = False
        def forward(self, rgb, prior, **kwargs):
            return self.core(prior, rgb)

    adapter = M2Adapter(model)

    # 1. Evaluate on DEV partition
    print("\nEvaluating M2-FINAL on independent DEV partition (314 records)...")
    dev_ds = M3MultiDomainDataset(
        manifest_path="data/m3/manifests/all_records.jsonl",
        split_records_path="data/m3/manifests/dev.txt",
        crop_size=512,
        is_training=False,
        height_aware_sampling=False,
    )
    dev_loader = DataLoader(dev_ds, batch_size=4, shuffle=False, num_workers=0)
    dev_results = evaluate_model_on_dataset(adapter, dev_loader, device)

    # 2. Evaluate on External NYC (500 records)
    print("\nEvaluating M2-FINAL on unseen NYC external test partition (500 records)...")
    nyc_ds = M3MultiDomainDataset(
        manifest_path="data/m3/manifests/all_records.jsonl",
        split_records_path="data/m3/manifests/nyc_external.txt",
        crop_size=512,
        is_training=False,
        height_aware_sampling=False,
    )
    nyc_loader = DataLoader(nyc_ds, batch_size=4, shuffle=False, num_workers=0)
    nyc_results = evaluate_model_on_dataset(adapter, nyc_loader, device)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "model": "M2-FINAL",
        "checkpoint": str(M2_CHECKPOINT_PATH),
        "dev_evaluation": dev_results,
        "nyc_evaluation": nyc_results
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print("\n" + "=" * 70)
    print("M2-FINAL BASELINE RESULTS:")
    print(f"  DEV Overall MAE:   {dev_results['global']['mae']:.4f} m")
    print(f"  DEV Overall RMSE:  {dev_results['global']['rmse']:.4f} m")
    print(f"  DEV Overall R2:    {dev_results['global']['r2']:.4f}")
    print(f"  DEV 0-2m MAE:      {dev_results['global']['height_buckets']['0_2m']['mae']:.4f} m")
    print(f"  DEV 2-10m MAE:     {dev_results['global']['height_buckets']['2_10m']['mae']:.4f} m")
    print(f"  DEV 10-20m MAE:    {dev_results['global']['height_buckets']['10_20m']['mae']:.4f} m")
    print(f"  DEV 20-50m MAE:    {dev_results['global']['height_buckets']['20_50m']['mae']:.4f} m")
    print(f"  DEV Selection Score: {dev_results['global']['selection_score']:.4f}")
    print("-" * 70)
    print(f"  NYC Overall MAE:   {nyc_results['global']['mae']:.4f} m")
    print(f"  NYC Overall RMSE:  {nyc_results['global']['rmse']:.4f} m")
    print(f"  NYC Overall R2:    {nyc_results['global']['r2']:.4f}")
    print(f"  NYC 10-20m MAE:    {nyc_results['global']['height_buckets']['10_20m']['mae']:.4f} m")
    print(f"  NYC 20-50m MAE:    {nyc_results['global']['height_buckets']['20_50m']['mae']:.4f} m")
    print(f"  NYC >=50m MAE:     {nyc_results['global']['height_buckets']['ge_50m']['mae']:.4f} m")
    print("=" * 70)
    print(f"Saved baseline evidence to: {OUT_FILE}")


if __name__ == "__main__":
    main()
