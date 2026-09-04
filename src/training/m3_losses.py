"""
DepthWizard (SIH26175) — M3 Multi-Objective Loss Functions
Author: DepthWizard Phase 2 Pipeline

Loss components:
1. Masked Smooth L1: Linear for large errors, quadratic near zero
2. Log1p Smooth L1: Relative percentage error penalty across low and tall structures
3. Gradient / Boundary Loss: Preserves sharp vertical building wall edges
4. Capped Height-Weighted Loss: Weights tall structures (10-20m, 20-50m, >=50m) to counter regression shrinkage
5. Asymmetric Underprediction Bias Penalty: Penalizes underpredicting tall structures (y > 15m)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (mask > 0) & torch.isfinite(target) & torch.isfinite(pred) & (target >= 0.0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )
        return loss[valid].mean()


class MaskedLog1pLoss(nn.Module):
    """
    Log1p Smooth L1 Loss: penalizes relative errors in log scale.
    Smooths scale discrepancies between low-rise (<5m) and high-rise (>20m).
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (mask > 0) & torch.isfinite(target) & torch.isfinite(pred) & (target >= 0.0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_clamped = torch.clamp(pred, min=0.0)
        log_pred = torch.log1p(pred_clamped)
        log_target = torch.log1p(target)

        diff = torch.abs(log_pred - log_target)
        loss = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )
        return loss[valid].mean()


class MaskedGradientLoss(nn.Module):
    """
    Spatial gradient / edge consistency loss.
    Enforces sharp vertical building wall boundaries.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_x = (mask[:, :, :, :-1] > 0) & (mask[:, :, :, 1:] > 0)
        valid_y = (mask[:, :, :-1, :] > 0) & (mask[:, :, 1:, :] > 0)

        grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        diff_x = torch.abs(grad_pred_x - grad_target_x)
        diff_y = torch.abs(grad_pred_y - grad_target_y)

        loss_x = diff_x[valid_x].mean() if valid_x.sum() > 0 else torch.tensor(0.0, device=pred.device)
        loss_y = diff_y[valid_y].mean() if valid_y.sum() > 0 else torch.tensor(0.0, device=pred.device)

        return 0.5 * (loss_x + loss_y)


class CappedHeightWeightedLoss(nn.Module):
    """
    Height-weighted regression loss:
    Weights each pixel by w = clip(1.0 + alpha * target, 1.0, max_weight).
    Directly counteracts the 56% ground / 0.06% tall structure skew.
    """
    def __init__(self, alpha: float = 0.08, max_weight: float = 4.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.max_weight = max_weight
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (mask > 0) & torch.isfinite(target) & torch.isfinite(pred) & (target >= 0.0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff = torch.abs(pred - target)
        loss_base = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )

        weights = torch.clamp(1.0 + self.alpha * target, min=1.0, max=self.max_weight)
        weighted_loss = loss_base * weights
        return weighted_loss[valid].mean()


class AsymmetricTallBiasLoss(nn.Module):
    """
    Penalizes underpredicting tall structures (target > height_thresh).
    Forces model not to collapse tall buildings into flat surfaces.
    """
    def __init__(self, height_thresh: float = 15.0):
        super().__init__()
        self.height_thresh = height_thresh

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Only evaluate on tall structures
        tall_mask = (mask > 0) & (target >= self.height_thresh)
        if tall_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        underpred = F.relu(target - pred)
        return (underpred[tall_mask] ** 2).mean()


class AsymmetricCanopyBiasLoss(nn.Module):
    """
    Penalizes underpredicting natural tree canopies and vegetation (4m <= target <= 20m).
    Directly counteracts the -5.47m negative bias on forest canopies without dominating building loss.
    """
    def __init__(self, min_height: float = 4.0, max_height: float = 20.0):
        super().__init__()
        self.min_height = min_height
        self.max_height = max_height

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        canopy_mask = (mask > 0) & (target >= self.min_height) & (target <= self.max_height)
        if canopy_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        underpred = F.relu(target - pred)
        return (underpred[canopy_mask] ** 2).mean()


class M3CompositeLoss(nn.Module):
    def __init__(
        self,
        lambda_smooth: float = 1.0,
        lambda_log: float = 0.20,
        lambda_grad: float = 0.15,
        lambda_height_weight: float = 0.35,
        lambda_tall_bias: float = 0.05,
        lambda_canopy_bias: float = 0.0,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_log = lambda_log
        self.lambda_grad = lambda_grad
        self.lambda_height_weight = lambda_height_weight
        self.lambda_tall_bias = lambda_tall_bias
        self.lambda_canopy_bias = lambda_canopy_bias

        self.smooth_l1 = MaskedSmoothL1Loss(beta=1.0)
        self.log_loss = MaskedLog1pLoss(beta=0.1)
        self.grad_loss = MaskedGradientLoss()
        self.height_loss = CappedHeightWeightedLoss(alpha=0.08, max_weight=4.0)
        self.tall_bias_loss = AsymmetricTallBiasLoss(height_thresh=15.0)
        self.canopy_bias_loss = AsymmetricCanopyBiasLoss(min_height=4.0, max_height=20.0)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_smooth = self.smooth_l1(pred, target, mask)
        l_log = self.log_loss(pred, target, mask) if self.lambda_log > 0 else torch.tensor(0.0, device=pred.device)
        l_grad = self.grad_loss(pred, target, mask) if self.lambda_grad > 0 else torch.tensor(0.0, device=pred.device)
        l_height = self.height_loss(pred, target, mask) if self.lambda_height_weight > 0 else torch.tensor(0.0, device=pred.device)
        l_tall = self.tall_bias_loss(pred, target, mask) if self.lambda_tall_bias > 0 else torch.tensor(0.0, device=pred.device)
        l_canopy = self.canopy_bias_loss(pred, target, mask) if self.lambda_canopy_bias > 0 else torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.lambda_smooth * l_smooth
            + self.lambda_log * l_log
            + self.lambda_grad * l_grad
            + self.lambda_height_weight * l_height
            + self.lambda_tall_bias * l_tall
            + self.lambda_canopy_bias * l_canopy
        )

        breakdown = {
            "loss_total": float(total_loss.item()),
            "loss_smooth": float(l_smooth.item()),
            "loss_log": float(l_log.item()),
            "loss_grad": float(l_grad.item()),
            "loss_height": float(l_height.item()),
            "loss_tall": float(l_tall.item()),
            "loss_canopy": float(l_canopy.item()),
        }

        return total_loss, breakdown
