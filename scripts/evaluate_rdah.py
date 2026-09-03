"""
DepthWizard (SIH26175) — Standalone Model Evaluation & Benchmark Script
Player 1: AI/ML Lead

Usage:
  python scripts/evaluate_rdah.py --checkpoint outputs/player1_stage_a/a1_rdah/checkpoints/best_mae_model.pth --manifest data/gamus/splits/stage_a1_val.txt --model_type rdah
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.rdah_net import RDAHNetCore
from src.models.rgb_only_baseline import RGBOnlyHeightNet
from src.training.dataset import GAMUSStageADataset
from src.training.metrics import compute_height_metrics, CLASS_NAMES, HEIGHT_BUCKETS

def evaluate_model(checkpoint_path, manifest_path, model_type="rdah", data_dir="data/gamus/dataset",
                   cache_dir="data/gamus/cache_dav2", out_dir="outputs/evaluations", device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 80)
    print("STANDALONE METRIC HEIGHT MODEL EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Manifest:   {manifest_path}")
    print(f"Model Type: {model_type.upper()} | Device: {device.upper()}")
    print("=" * 80)

    # 1. Dataset
    dataset = GAMUSStageADataset(
        manifest_path=manifest_path,
        data_dir=data_dir,
        dav2_cache_dir=cache_dir,
        crop_size=(512, 512),
        is_training=False,
        transform_geo=False
    )
    print(f"Loaded {len(dataset)} validation samples.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Model
    if model_type == "rdah":
        model = RDAHNetCore(d_model=32, num_heads=4)
    elif model_type == "rgb_only":
        model = RGBOnlyHeightNet(d_model=32, num_heads=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}! Using random weights.")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []
    all_semantics = []
    sample_metrics_list = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            sid = batch['sample_id'][0]
            rgb = batch['rgb'].to(device)
            hgt = batch['height'].to(device)
            mask = batch['valid_mask'].to(device)
            depth = batch['rel_depth'].to(device)
            sem = batch['semantic'].to(device)

            if model_type == "rdah":
                pred = model(depth, rgb)
            else:
                pred = model(rgb)

            p_np = pred.squeeze().cpu().numpy().astype(np.float32)
            t_np = hgt.squeeze().cpu().numpy().astype(np.float32)
            m_np = mask.squeeze().cpu().numpy()
            s_np = sem.squeeze().cpu().numpy()
            rgb_np = np.transpose(rgb.squeeze().cpu().numpy(), (1, 2, 0))

            all_preds.append(p_np)
            all_targets.append(t_np)
            all_masks.append(m_np)
            all_semantics.append(s_np)

            s_metric = compute_height_metrics(p_np, t_np, m_np, s_np)
            sample_metrics_list.append({
                'sample_id': sid,
                'mae_m': s_metric['mae_m'],
                'rmse_m': s_metric['rmse_m'],
                'pearson_r': s_metric['pearson_r'],
                'spearman_rho': s_metric['spearman_rho']
            })

            # Save qualitative visual composites for top samples
            if idx < 10 or idx % 20 == 0:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
                axes[0].imshow(np.clip(rgb_np, 0.0, 1.0))
                axes[0].set_title(f"Optical RGB\nSample: {sid}", fontsize=10, fontweight='bold')
                axes[0].axis('off')

                t_disp = np.copy(t_np)
                t_disp[~m_np] = np.nan
                vmax = max(20.0, np.percentile(t_np[m_np], 98) if m_np.sum()>0 else 20.0)
                im1 = axes[1].imshow(t_disp, cmap='turbo', vmin=0, vmax=vmax)
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label('Height (m)', fontsize=8)
                axes[1].set_title(f"Ground Truth AGL\nMean: {np.mean(t_np[m_np]):.1f}m", fontsize=10, fontweight='bold')
                axes[1].axis('off')

                p_disp = np.copy(p_np)
                p_disp[~m_np] = np.nan
                im2 = axes[2].imshow(p_disp, cmap='turbo', vmin=0, vmax=vmax)
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label('Height (m)', fontsize=8)
                axes[2].set_title(f"{model_type.upper()} Predicted AGL\nMean: {np.mean(p_np[m_np]):.1f}m", fontsize=10, fontweight='bold')
                axes[2].axis('off')

                err_map = np.abs(p_np - t_np)
                err_map[~m_np] = np.nan
                im3 = axes[3].imshow(err_map, cmap='magma', vmin=0, vmax=10.0)
                fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04).set_label('Abs Error (m)', fontsize=8)
                axes[3].set_title(f"Absolute Error Map\nMAE: {s_metric['mae_m']:.2f} m", fontsize=10, fontweight='bold')
                axes[3].axis('off')

                fig.suptitle(f"DepthWizard Metric Height Estimation | {model_type.upper()} | {sid}", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{sid}_{model_type}_eval_vis.png"), bbox_inches='tight')
                plt.close(fig)

    all_p = np.stack(all_preds, axis=0)
    all_t = np.stack(all_targets, axis=0)
    all_m = np.stack(all_masks, axis=0)
    all_s = np.stack(all_semantics, axis=0)

    global_metrics = compute_height_metrics(all_p, all_t, all_m, all_s)

    eval_summary = {
        'checkpoint_path': checkpoint_path,
        'manifest_path': manifest_path,
        'model_type': model_type,
        'total_evaluated_samples': len(dataset),
        'global_metrics': global_metrics,
        'per_sample_metrics': sample_metrics_list
    }

    out_json = os.path.join(out_dir, f"{model_type}_evaluation_summary.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS ({model_type.upper()} on {len(dataset)} samples):")
    print(f"  Global MAE:        {global_metrics['mae_m']:.3f} m")
    print(f"  Global RMSE:       {global_metrics['rmse_m']:.3f} m")
    print(f"  Pearson r:         {global_metrics['pearson_r']:+.3f}")
    print(f"  Spearman rho:      {global_metrics['spearman_rho']:+.3f}")
    print("\n  Class-wise MAE (m):")
    for cname, cmae in global_metrics['class_mae'].items():
        if cmae is not None:
            print(f"    {cname:<16}: {cmae:.3f} m")
    print("\n  Height-bucket MAE (m):")
    for bname, bmae in global_metrics['bucket_mae'].items():
        if bmae is not None:
            print(f"    {bname:<16}: {bmae:.3f} m")
    print("=" * 80)
    print(f"Saved evaluation summary: {out_json}")
    return eval_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate DepthWizard Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default="data/gamus/splits/stage_a1_val.txt", help="Path to manifest")
    parser.add_argument("--model_type", type=str, default="rdah", choices=["rdah", "rgb_only"])
    parser.add_argument("--data_dir", type=str, default="data/gamus/dataset")
    parser.add_argument("--cache_dir", type=str, default="data/gamus/cache_dav2")
    parser.add_argument("--out_dir", type=str, default="outputs/evaluations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_model(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        model_type=args.model_type,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        device=device
    )

if __name__ == '__main__':
    main()
