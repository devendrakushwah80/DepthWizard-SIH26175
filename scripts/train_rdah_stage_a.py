"""
DepthWizard (SIH26175) — Master Stage A Height Estimation Training Script
Player 1: AI/ML Lead

Usage:
  python scripts/train_rdah_stage_a.py --config configs/rdah_stage_a0.yaml --model_type rdah
  python scripts/train_rdah_stage_a.py --config configs/rdah_stage_a0.yaml --model_type rgb_only
  python scripts/train_rdah_stage_a.py --config configs/rdah_stage_a1.yaml --model_type rdah
"""

import os
import sys
import argparse
import random
import yaml
import json
import torch
import numpy as np

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.rdah_net import RDAHNetCore
from src.models.rgb_only_baseline import RGBOnlyHeightNet
from src.training.dataset import GAMUSStageADataset
from src.training.trainer import DepthWizardTrainer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Train DepthWizard Stage A Height Estimation Models")
    parser.add_argument("--config", type=str, default="configs/rdah_stage_a0.yaml", help="Path to config YAML")
    parser.add_argument("--model_type", type=str, default="rdah", choices=["rdah", "rgb_only"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['epochs'] = args.epochs

    set_seed(config.get('random_seed', 42))

    stage = config.get('stage', 'A0')
    out_dir = f"outputs/player1_stage_a/{stage.lower()}_{args.model_type}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"DEPTHWIZARD STAGE {stage} HEIGHT ESTIMATION TRAINING")
    print(f"Model Type: {args.model_type.upper()} | Target Device: {args.device.upper()}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Datasets
    manifest_train = config['manifests']['train']
    manifest_val = config['manifests']['val']
    data_dir = config['data']['data_dir']
    cache_dir = config['data'].get('cache_dir', None)
    crop_size = tuple(config['data'].get('crop_size', [512, 512]))

    print("\n--- Loading Datasets ---")
    train_ds = GAMUSStageADataset(
        manifest_path=manifest_train,
        data_dir=data_dir,
        dav2_cache_dir=cache_dir,
        crop_size=crop_size,
        is_training=True,
        transform_geo=True
    )
    val_ds = GAMUSStageADataset(
        manifest_path=manifest_val,
        data_dir=data_dir,
        dav2_cache_dir=cache_dir,
        crop_size=crop_size,
        is_training=False,
        transform_geo=False
    )
    print(f"Train Dataset: {len(train_ds)} samples from {manifest_train}")
    print(f"Val Dataset:   {len(val_ds)} samples from {manifest_val}")

    if len(train_ds) == 0:
        print("ERROR: Training dataset has 0 samples! Please run download_stage_a_subset.py first.")
        sys.exit(1)

    # 2. Model Initialization
    d_model = config.get('model', {}).get('d_model', 32)
    num_heads = config.get('model', {}).get('num_heads', 4)

    if args.model_type == "rdah":
        model = RDAHNetCore(d_model=d_model, num_heads=num_heads)
        model_name = "RDAH-Net (RGB + Frozen DAV2 Prior)"
    elif args.model_type == "rgb_only":
        model = RGBOnlyHeightNet(d_model=d_model, num_heads=num_heads)
        model_name = "RGB-Only Baseline HeightNet (No Depth Prior)"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_name}")
    print(f"Total Parameters:     {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M)")

    config['name'] = f"{stage}_{args.model_type.upper()}"

    # 3. Trainer
    trainer = DepthWizardTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        out_dir=out_dir,
        device=args.device
    )

    # 4. Train
    num_epochs = config.get('epochs', 10)
    history = trainer.train(num_epochs=num_epochs)

    # 5. Final Comprehensive Evaluation with Best Checkpoint
    best_ckpt = os.path.join(out_dir, "checkpoints", "best_mae_model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=args.device))
        print(f"\nLoaded best MAE model checkpoint from: {best_ckpt}")

    final_metrics = trainer.evaluate(epoch=num_epochs, save_visuals=True)
    
    summary_results = {
        'stage': stage,
        'model_type': args.model_type,
        'model_name': model_name,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'epochs_trained': num_epochs,
        'best_val_mae_m': trainer.best_mae,
        'best_val_rmse_m': trainer.best_rmse,
        'final_val_metrics': final_metrics,
        'config': config
    }

    results_path = os.path.join(out_dir, "final_summary_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved final summary metrics: {results_path}")

if __name__ == '__main__':
    main()
