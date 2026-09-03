"""
DepthWizard (SIH26175) — Unified Height Estimation Trainer & Evaluator
Player 1: AI/ML Lead
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import RDAHCombinedLoss
from src.training.metrics import compute_height_metrics

class DepthWizardTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, out_dir, device="cuda"):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.out_dir = out_dir
        self.device = device
        
        os.makedirs(out_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        self.plots_dir = os.path.join(out_dir, "plots")
        self.vis_dir = os.path.join(out_dir, "visualizations")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.model.to(self.device)

        # Optimizer
        opt_cfg = config.get('optimizer', {})
        lr = float(opt_cfg.get('lr', 1e-4))
        wd = float(opt_cfg.get('weight_decay', 1e-4))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

        # Scheduler
        epochs = config.get('epochs', 20)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        # Loss Function
        loss_cfg = config.get('loss', {})
        beta = float(loss_cfg.get('beta', 1.0))
        grad_w = float(loss_cfg.get('grad_weight', 0.5))
        self.criterion = RDAHCombinedLoss(beta=beta, grad_weight=grad_w).to(self.device)

        # AMP Scaler
        self.use_amp = (device == "cuda") and config.get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.batch_size = config.get('batch_size', 2)
        self.grad_accum = config.get('gradient_accumulation_steps', 4)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(device=="cuda")
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=(device=="cuda")
        )

        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_smooth_l1': [],
            'train_grad_loss': [],
            'val_mae_m': [],
            'val_rmse_m': [],
            'val_pearson_r': [],
            'val_spearman_rho': [],
            'lr': [],
            'epoch_time_s': []
        }

        self.best_mae = float('inf')
        self.best_rmse = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_grad = 0.0
        self.optimizer.zero_grad()

        t0 = time.time()
        num_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            rgb = batch['rgb'].to(self.device)
            hgt = batch['height'].to(self.device)
            mask = batch['valid_mask'].to(self.device)
            depth = batch['rel_depth'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Check model signature (RDAHNet vs RGBOnly)
                if hasattr(self.model, 'rdah_core'):
                    pred = self.model.rdah_core(depth, rgb)
                elif hasattr(self.model, 'img_encoder') and not hasattr(self.model, 'depth_encoder'):
                    pred = self.model(rgb) # RGB-only
                else:
                    pred = self.model(depth, rgb)

                loss, l1, l_grad = self.criterion(pred, hgt, mask)
                loss_scaled = loss / self.grad_accum

            self.scaler.scale(loss_scaled).backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            total_l1 += l1.item()
            total_grad += l_grad.item()

        avg_loss = total_loss / max(1, num_batches)
        avg_l1 = total_l1 / max(1, num_batches)
        avg_grad = total_grad / max(1, num_batches)
        epoch_time = time.time() - t0

        return avg_loss, avg_l1, avg_grad, epoch_time

    def evaluate(self, epoch=None, save_visuals=False):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        all_semantics = []

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                sid = batch['sample_id'][0]
                rgb = batch['rgb'].to(self.device)
                hgt = batch['height'].to(self.device)
                mask = batch['valid_mask'].to(self.device)
                depth = batch['rel_depth'].to(self.device)
                sem = batch['semantic'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if hasattr(self.model, 'rdah_core'):
                        pred = self.model.rdah_core(depth, rgb)
                    elif hasattr(self.model, 'img_encoder') and not hasattr(self.model, 'depth_encoder'):
                        pred = self.model(rgb)
                    else:
                        pred = self.model(depth, rgb)

                pred_np = pred.squeeze().cpu().numpy().astype(np.float32)
                hgt_np = hgt.squeeze().cpu().numpy().astype(np.float32)
                mask_np = mask.squeeze().cpu().numpy()
                sem_np = sem.squeeze().cpu().numpy()
                rgb_np = np.transpose(rgb.squeeze().cpu().numpy(), (1, 2, 0))

                all_preds.append(pred_np)
                all_targets.append(hgt_np)
                all_masks.append(mask_np)
                all_semantics.append(sem_np)

                if save_visuals and (idx < 6 or idx % 10 == 0):
                    self._save_prediction_vis(sid, rgb_np, hgt_np, pred_np, mask_np, epoch)

        # Aggregate metrics
        all_p = np.stack(all_preds, axis=0)
        all_t = np.stack(all_targets, axis=0)
        all_m = np.stack(all_masks, axis=0)
        all_s = np.stack(all_semantics, axis=0)

        metrics = compute_height_metrics(all_p, all_t, all_m, all_s)
        return metrics

    def _save_prediction_vis(self, sid, rgb, hgt, pred, mask, epoch):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        
        # 1. RGB
        axes[0].imshow(np.clip(rgb, 0.0, 1.0))
        axes[0].set_title(f"Optical RGB\nSample: {sid}", fontsize=10, fontweight='bold')
        axes[0].axis('off')

        # 2. GT AGL
        h_disp = np.copy(hgt)
        h_disp[~mask] = np.nan
        vmax = max(20.0, np.percentile(hgt[mask], 98) if mask.sum()>0 else 20.0)
        im1 = axes[1].imshow(h_disp, cmap='turbo', vmin=0, vmax=vmax)
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label('Height (m)', fontsize=8)
        axes[1].set_title(f"Ground Truth AGL\nMean: {np.mean(hgt[mask]):.1f}m", fontsize=10, fontweight='bold')
        axes[1].axis('off')

        # 3. Predicted AGL
        p_disp = np.copy(pred)
        p_disp[~mask] = np.nan
        im2 = axes[2].imshow(p_disp, cmap='turbo', vmin=0, vmax=vmax)
        cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_label('Height (m)', fontsize=8)
        axes[2].set_title(f"RDAH-Net Prediction\nMean: {np.mean(pred[mask]):.1f}m", fontsize=10, fontweight='bold')
        axes[2].axis('off')

        # 4. Error Map
        err_map = np.abs(pred - hgt)
        err_map[~mask] = np.nan
        im3 = axes[3].imshow(err_map, cmap='magma', vmin=0, vmax=10.0)
        cbar3 = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        cbar3.set_label('Abs Error (m)', fontsize=8)
        sample_mae = np.mean(err_map[mask]) if mask.sum()>0 else 0.0
        axes[3].set_title(f"Absolute Error Map\nSample MAE: {sample_mae:.2f} m", fontsize=10, fontweight='bold')
        axes[3].axis('off')

        ep_str = f"Epoch {epoch}" if epoch is not None else "Evaluation"
        fig.suptitle(f"DepthWizard Metric Height Estimation | {ep_str} | Sample: {sid}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        vis_path = os.path.join(self.vis_dir, f"{sid}_ep{epoch if epoch is not None else 0}_vis.png")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close(fig)

    def train(self, num_epochs=10):
        print("=" * 80)
        print(f"STARTING TRAINING: {self.config.get('name', 'RDAH-Net')} ({num_epochs} Epochs on {self.device.upper()})")
        print(f"Train samples: {len(self.train_dataset)} | Val samples: {len(self.val_dataset)}")
        print("=" * 80)

        # Initial zero-epoch baseline evaluation
        val_m = self.evaluate(epoch=0, save_visuals=True)
        print(f"[Epoch 00/{num_epochs:02d} - Initial] Val MAE: {val_m['mae_m']:.3f} m | RMSE: {val_m['rmse_m']:.3f} m | Pearson r: {val_m['pearson_r']:+.3f}")

        for ep in range(1, num_epochs + 1):
            loss, l1, grad, dur = self.train_epoch(ep)
            self.scheduler.step()
            curr_lr = self.scheduler.get_last_lr()[0]

            # Validation
            val_metrics = self.evaluate(epoch=ep, save_visuals=(ep % 5 == 0 or ep == num_epochs))
            mae = val_metrics['mae_m']
            rmse = val_metrics['rmse_m']
            pr = val_metrics['pearson_r']
            rho = val_metrics['spearman_rho']

            # Record history
            self.history['epoch'].append(ep)
            self.history['train_loss'].append(loss)
            self.history['train_smooth_l1'].append(l1)
            self.history['train_grad_loss'].append(grad)
            self.history['val_mae_m'].append(mae)
            self.history['val_rmse_m'].append(rmse)
            self.history['val_pearson_r'].append(pr)
            self.history['val_spearman_rho'].append(rho)
            self.history['lr'].append(curr_lr)
            self.history['epoch_time_s'].append(dur)

            # Checkpoint management
            if mae < self.best_mae:
                self.best_mae = mae
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_mae_model.pth"))
                is_best_mae = True
            else:
                is_best_mae = False

            if rmse < self.best_rmse:
                self.best_rmse = rmse
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_rmse_model.pth"))
                is_best_rmse = True
            else:
                is_best_rmse = False

            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "latest_model.pth"))

            star = " (*BEST MAE*)" if is_best_mae else (" (*BEST RMSE*)" if is_best_rmse else "")
            print(f"[Epoch {ep:02d}/{num_epochs:02d}] Loss: {loss:.4f} (L1: {l1:.4f}, Grad: {grad:.4f}) | Val MAE: {mae:.3f} m | RMSE: {rmse:.3f} m | Pearson: {pr:+.3f} | LR: {curr_lr:.2e} | Time: {dur:.1f}s{star}")

        # Save History JSON
        hist_path = os.path.join(self.out_dir, "training_history.json")
        with open(hist_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

        # Plot Curves
        self._plot_training_curves()

        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETED: Best Val MAE: {self.best_mae:.3f} m | Best Val RMSE: {self.best_rmse:.3f} m")
        print("=" * 80)
        return self.history

    def _plot_training_curves(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        epochs = self.history['epoch']
        
        # 1. Training Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-o', label='Total Loss', linewidth=2)
        axes[0].plot(epochs, self.history['train_smooth_l1'], 'g--', label='Smooth L1', alpha=0.7)
        axes[0].plot(epochs, self.history['train_grad_loss'], 'r--', label='Gradient Loss', alpha=0.7)
        axes[0].set_title("Training Loss Convergence", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend()

        # 2. Validation MAE & RMSE
        axes[1].plot(epochs, self.history['val_mae_m'], 'm-o', label='Val MAE (m)', linewidth=2)
        axes[1].plot(epochs, self.history['val_rmse_m'], 'c-s', label='Val RMSE (m)', linewidth=2)
        axes[1].set_title("Validation Physical Accuracy (Metres)", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Error (m)")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

        # 3. Pearson Correlation
        axes[2].plot(epochs, self.history['val_pearson_r'], 'k-^', label='Pearson r', linewidth=2)
        axes[2].plot(epochs, self.history['val_spearman_rho'], 'y-d', label='Spearman ρ', linewidth=2)
        axes[2].set_title("Validation Structural Correlation", fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Correlation")
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend()

        plt.suptitle(f"{self.config.get('name', 'Model')} Training Curves", fontsize=13, fontweight='bold')
        plt.tight_layout()
        curve_path = os.path.join(self.plots_dir, "training_curves.png")
        plt.savefig(curve_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved training curves: {curve_path}")
