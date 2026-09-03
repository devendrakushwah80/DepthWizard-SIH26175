"""
DepthWizard (SIH26175) — Metric Height Loss Functions
Player 1: AI/ML Lead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSmoothL1Loss(nn.Module):
    """
    Masked Smooth L1 (Huber) Loss with valid_mask handling.
    Smooth L1 is linear for large residuals (outlier-robust) and quadratic near 0.
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)
            mask: (B, 1, H, W) bool or float
        """
        if mask is not None:
            valid = (mask > 0) & torch.isfinite(target) & torch.isfinite(pred) & (target >= 0.0)
        else:
            valid = torch.isfinite(target) & torch.isfinite(pred) & (target >= 0.0)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )
        return loss[valid].mean()

class MaskedGradientLoss(nn.Module):
    """
    Masked Spatial Gradient / Edge Loss.
    Penalizes difference in image gradients (Sobel / finite differences) along x and y
    to preserve sharp vertical building wall boundaries.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        if mask is not None:
            valid_x = (mask[:, :, :, :-1] > 0) & (mask[:, :, :, 1:] > 0)
            valid_y = (mask[:, :, :-1, :] > 0) & (mask[:, :, 1:, :] > 0)
        else:
            valid_x = torch.ones_like(pred[:, :, :, :-1], dtype=torch.bool)
            valid_y = torch.ones_like(pred[:, :, :-1, :], dtype=torch.bool)

        # Finite differences
        grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        grad_tgt_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_tgt_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        diff_x = torch.abs(grad_pred_x - grad_tgt_x)
        diff_y = torch.abs(grad_pred_y - grad_tgt_y)

        loss_x = diff_x[valid_x].mean() if valid_x.sum() > 0 else torch.tensor(0.0, device=pred.device)
        loss_y = diff_y[valid_y].mean() if valid_y.sum() > 0 else torch.tensor(0.0, device=pred.device)

        return loss_x + loss_y

class RDAHCombinedLoss(nn.Module):
    """
    Combined Stage A Loss:
    L = MaskedSmoothL1(beta=1.0) + grad_weight * MaskedGradientLoss()
    """
    def __init__(self, beta=1.0, grad_weight=0.5):
        super().__init__()
        self.smooth_l1 = MaskedSmoothL1Loss(beta=beta)
        self.grad_loss = MaskedGradientLoss()
        self.grad_weight = grad_weight

    def forward(self, pred, target, mask=None):
        l_smooth = self.smooth_l1(pred, target, mask)
        if self.grad_weight > 0:
            l_grad = self.grad_loss(pred, target, mask)
            total_loss = l_smooth + self.grad_weight * l_grad
            return total_loss, l_smooth, l_grad
        else:
            return l_smooth, l_smooth, torch.tensor(0.0, device=pred.device)


class HeightAwareCombinedLoss(nn.Module):
    """Bounded height/semantic-aware Stage A2 regression objective.

    Pixel weights are divided by their valid-pixel mean.  Consequently a trial
    cannot win merely by changing the overall loss scale, and the configured
    weights only redistribute contribution between target regimes.
    """

    def __init__(
        self,
        beta=1.0,
        grad_weight=0.5,
        log_weight=0.0,
        height_weights=(1.0, 1.0, 1.5, 2.5, 4.0),
        building_weight=1.0,
        building_class_id=3,
    ):
        super().__init__()
        if len(height_weights) != 5:
            raise ValueError("height_weights must contain five bucket weights")
        self.beta = float(beta)
        self.grad_weight = float(grad_weight)
        self.log_weight = float(log_weight)
        self.register_buffer(
            "height_weights", torch.tensor(height_weights, dtype=torch.float32)
        )
        self.register_buffer(
            "height_boundaries", torch.tensor([2.0, 10.0, 20.0, 50.0], dtype=torch.float32)
        )
        self.building_weight = float(building_weight)
        self.building_class_id = int(building_class_id)
        self.grad_loss = MaskedGradientLoss()

    def _pixel_weights(self, target, semantic, valid):
        # Invalid/nodata targets (for example GAMUS -5 m) must not enter
        # bucket or nonlinear loss arithmetic.  Their eventual zero gradient
        # is insufficient because 0 * NaN is still NaN in backward passes.
        safe_target = torch.where(torch.isfinite(target) & (target >= 0.0), target, 0.0)
        bucket = torch.bucketize(safe_target, self.height_boundaries)
        weights = self.height_weights[bucket]
        if semantic is not None and self.building_weight != 1.0:
            if semantic.ndim == 3:
                semantic = semantic.unsqueeze(1)
            weights = weights * torch.where(
                semantic == self.building_class_id,
                self.building_weight,
                1.0,
            )
        mean_weight = weights[valid].mean().clamp_min(1e-6)
        return weights / mean_weight

    def _weighted_smooth_l1(self, pred, target, weights, valid):
        # Select valid values before any arithmetic.  Masking a tensor after
        # evaluating an invalid branch can leave NaNs in its autograd graph.
        pred = pred[valid]
        target = target[valid]
        weights = weights[valid]
        diff = torch.abs(pred - target)
        raw = torch.where(
            diff < self.beta,
            0.5 * diff.square() / self.beta,
            diff - 0.5 * self.beta,
        )
        return (raw * weights).mean()

    def forward(self, pred, target, mask=None, semantic=None):
        valid = torch.isfinite(target) & torch.isfinite(pred) & (target >= 0.0)
        if mask is not None:
            valid = valid & (mask > 0)
        if not torch.any(valid):
            zero = pred.sum() * 0.0
            return zero, {"metric": zero, "gradient": zero, "logheight": zero}

        weights = self._pixel_weights(target, semantic, valid)
        metric = self._weighted_smooth_l1(pred, target, weights, valid)
        if self.grad_weight:
            safe_pred = torch.where(valid, pred, 0.0)
            safe_target = torch.where(valid, target, 0.0)
            gradient = self.grad_loss(safe_pred, safe_target, valid)
        else:
            gradient = pred.sum() * 0.0

        if self.log_weight:
            # GAMUS uses -5 m nodata values.  Apply the valid selection before
            # log1p so these pixels never create a hidden NaN gradient.
            valid_pred = pred[valid]
            valid_target = target[valid]
            valid_weights = weights[valid]
            pred_log = torch.log1p(torch.clamp_min(valid_pred, 0.0))
            target_log = torch.log1p(valid_target)
            diff = torch.abs(pred_log - target_log)
            raw = torch.where(
                diff < self.beta,
                0.5 * diff.square() / self.beta,
                diff - 0.5 * self.beta,
            )
            logheight = (raw * valid_weights).mean()
        else:
            logheight = pred.sum() * 0.0

        total = metric + self.grad_weight * gradient + self.log_weight * logheight
        return total, {
            "metric": metric,
            "gradient": gradient,
            "logheight": logheight,
        }
