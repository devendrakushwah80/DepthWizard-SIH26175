"""
DepthWizard (SIH26175) — M3-D Natural Domain Hardening Trainer
Trains M3-D on mixed urban (GAMUS + US3D) and natural (US3D LiDAR nDSM) data
with canopy-aware loss and sampling, initialized from M3-FINAL weights.
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.m3_net import M3NetCore
from src.training.m3_dataset import M3MultiDomainDataset
from src.training.m3_losses import M3CompositeLoss
from scripts.evaluate_validated_natural import select_validated_natural_tiles, evaluate_batch

M3_FINAL_CKPT = PROJECT_ROOT / "models" / "m3_final" / "M3_FINAL.pth"
EXP_DIR = PROJECT_ROOT / "models" / "m3_experiments" / "M3-D_natural"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
METRICS_JSON = PROJECT_ROOT / "outputs" / "m3" / "M3-D_PROGRESS.json"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def train_m3d(
    epochs: int = 15,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 1.0e-4,
    lambda_canopy_bias: float = 0.12,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("DepthWizard (SIH26175) — Training M3-D Natural Domain Adaptation")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}x{grad_accum_steps} | LR: {lr}")
    print(f"Canopy Bias Penalty: {lambda_canopy_bias} | Checkpoints: {CHECKPOINTS_DIR}")
    print("=" * 75)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    train_ds = M3MultiDomainDataset(
        manifest_path=str(PROJECT_ROOT / "data" / "m3" / "manifests" / "all_records.jsonl"),
        split_records_path=str(PROJECT_ROOT / "data" / "m3" / "manifests" / "m3d_train.txt"),
        crop_size=512,
        is_training=True,
        height_aware_sampling=True,
        gsd_dropout_prob=0.15,
    )

    # Domain-balanced sampler: 35% natural, 65% urban
    weights = []
    nat_count = 0
    urb_count = 0
    for rec in train_ds.records:
        if "NATURAL" in rec.get("dataset", ""):
            nat_count += 1
        else:
            urb_count += 1

    p_nat = 0.35 / max(1, nat_count)
    p_urb = 0.65 / max(1, urb_count)
    for rec in train_ds.records:
        if "NATURAL" in rec.get("dataset", ""):
            weights.append(p_nat)
        else:
            weights.append(p_urb)

    num_samples_per_epoch = 1200
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples_per_epoch,
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    print(f"Training dataset loaded: {len(train_ds)} scenes ({nat_count} natural, {urb_count} urban)")
    print(f"Domain-balanced sampler: {num_samples_per_epoch} samples ({len(train_loader)} batches) per epoch (35% natural / 65% urban)")

    # 2. Model: Initialize from M3-FINAL weights
    print(f"Initializing M3-D from M3-FINAL weights ({M3_FINAL_CKPT})...")
    model = M3NetCore(enable_gsd_conditioning=True).to(device)
    m3_weights = torch.load(M3_FINAL_CKPT, map_location=device, weights_only=True)
    if isinstance(m3_weights, dict) and "model_state_dict" in m3_weights:
        m3_weights = m3_weights["model_state_dict"]
    model.load_state_dict(m3_weights)

    # 3. Loss with Canopy Bias Term
    criterion = M3CompositeLoss(
        lambda_smooth=1.0,
        lambda_log=0.20,
        lambda_grad=0.15,
        lambda_height_weight=0.35,
        lambda_tall_bias=0.05,
        lambda_canopy_bias=lambda_canopy_bias,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Validation reference models
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dav2_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    dav2_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
    from src.models.rdah_net import RDAHNetCore
    m2_dummy = RDAHNetCore().to(device).eval()
    val_models = (dav2_proc, dav2_model, m2_dummy, model)

    val_natural_tiles = select_validated_natural_tiles()

    best_forest_mae = 999.0
    best_overall_score = 999.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_canopy_loss = 0.0
        batch_count = 0

        optimizer.zero_grad(set_to_none=True)

        # Run training loop
        for b_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth_prior = batch["prior"].to(device, non_blocking=True)
            target = batch["agl"].to(device, non_blocking=True)
            valid_mask = batch["mask"].to(device, non_blocking=True)
            gsd_m = batch["gsd_m"].to(device, non_blocking=True)
            gsd_known = batch["gsd_known"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(rgb, depth_prior, gsd_m=gsd_m, gsd_known=gsd_known)
                loss, loss_dict = criterion(pred, target, valid_mask)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss_dict["loss_total"]
            running_canopy_loss += loss_dict.get("loss_canopy", 0.0)
            batch_count += 1

            if (b_idx + 1) % 150 == 0:
                print(f"  [Epoch {epoch:02d} | Batch {b_idx+1:04d}/{len(train_loader)}] Loss: {running_loss/batch_count:.4f} | Canopy Loss: {running_canopy_loss/batch_count:.4f}")

        scheduler.step()
        train_duration = time.time() - epoch_start
        avg_loss = running_loss / max(1, batch_count)

        # Evaluate on Validated Natural Benchmark (Forest Canopy)
        model.eval()
        with torch.inference_mode():
            nat_eval = evaluate_batch(val_models, val_natural_tiles["forest_canopy"])
            forest_stats = nat_eval["m3_final"]
            comb_eval = evaluate_batch(val_models, val_natural_tiles["forest_canopy"] + val_natural_tiles["mixed_vegetation"] + val_natural_tiles["bare_sloped_terrain"])
            comb_stats = comb_eval["m3_final"]

        print(f"\n[Epoch {epoch:02d} Summary] Duration: {train_duration:.1f}s | Train Loss: {avg_loss:.4f}")
        print(f"  Forest Canopy MAE: {forest_stats['mae']:.4f}m (M3-FINAL was 5.7619m) | Bias: {forest_stats['bias']:.4f}m (was -5.4672m)")
        print(f"  Combined Natural MAE: {comb_stats['mae']:.4f}m (M3-FINAL was 2.8899m) | Within 2m: {comb_stats['within_2m']}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "forest_mae": forest_stats["mae"],
            "forest_bias": forest_stats["bias"],
            "forest_rmse": forest_stats["rmse"],
            "combined_natural_mae": comb_stats["mae"],
            "within_2m": comb_stats["within_2m"],
        }
        history.append(epoch_record)

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "experiment_name": "M3-D_natural",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": epoch_record,
        }, CHECKPOINTS_DIR / "last.pth")

        # Save best forest checkpoint (ensuring combined MAE does not regress)
        if forest_stats["mae"] < best_forest_mae and comb_stats["mae"] <= 2.92:
            best_forest_mae = forest_stats["mae"]
            print(f"  >>> NEW BEST FOREST MAE: {best_forest_mae:.4f}m (Combined: {comb_stats['mae']:.4f}m)! Saving best_forest.pth")
            torch.save({
                "epoch": epoch,
                "experiment_name": "M3-D_natural",
                "model_state_dict": model.state_dict(),
                "metrics": epoch_record,
            }, CHECKPOINTS_DIR / "best_forest.pth")

        # Early stopping check: if forest MAE improved significantly and combined MAE is good
        if forest_stats["mae"] < 4.50 and comb_stats["mae"] <= 2.80:
            print(f"\n[EARLY STOPPING] Target forest MAE < 4.50m reached at epoch {epoch}. Terminating training loop.")
            break

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump({"history": history, "best_forest_mae": best_forest_mae}, f, indent=2)

    print("\n" + "=" * 75)
    print("M3-D Training Complete. Proceeding to Strict Promotion Gate Evaluation...")
    print("=" * 75)


if __name__ == "__main__":
    train_m3d(epochs=15, batch_size=4, grad_accum_steps=2, lr=6.0e-5, lambda_canopy_bias=0.06)
