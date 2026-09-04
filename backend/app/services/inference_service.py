"""
DepthWizard (SIH26175) — Height Inference Service
Player 4: Backend & Systems Integration Lead

Performs sliding-window metric height estimation with DAV2 zero-shot prior extraction
and 2D Hanning window overlap blending.
"""

import time
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from backend.app.services.model_service import model_service

class InferenceService:
    def __init__(self):
        self.crop_size = 512
        self.step_size = 256
        w1d = np.hanning(self.crop_size)
        self.window = np.clip(np.outer(w1d, w1d), 0.05, 1.0).astype(np.float32)

    def _window_positions(self, length: int) -> list[int]:
        """Return a fully covering, deterministic window grid for one axis."""
        if length <= self.crop_size:
            return [0]
        positions = list(range(0, length - self.crop_size + 1, self.step_size))
        final_position = length - self.crop_size
        if positions[-1] != final_position:
            positions.append(final_position)
        return positions

    @staticmethod
    def _synchronize(device: str):
        if device == "cuda":
            torch.cuda.synchronize()

    def predict_height_map(self, rgb_image: np.ndarray, gsd_m: Optional[float] = None) -> dict:
        """
        Runs full-resolution height estimation on RGB optical input (H, W, 3) [0..255 uint8 or float].
        Returns:
          {
            'predicted_agl_m': (H, W) float32 array in metres,
            'dav2_prior': (H, W) float32 array,
            'inference_time_s': float
          }
        """
        t0 = time.time()
        
        # Normalize RGB to [0..1] float32
        if rgb_image.dtype == np.uint8 or rgb_image.max() > 1.5:
            rgb_norm = rgb_image.astype(np.float32) / 255.0
            rgb_uint8 = rgb_image.astype(np.uint8)
        else:
            rgb_norm = rgb_image.astype(np.float32)
            rgb_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)

        if rgb_norm.ndim != 3 or rgb_norm.shape[2] != 3:
            raise ValueError("RGB input must have shape (height, width, 3)")
        if not np.isfinite(rgb_norm).all():
            raise ValueError("RGB input contains non-finite values")

        H, W, _ = rgb_norm.shape
        if H < 1 or W < 1:
            raise ValueError("RGB input dimensions must be non-zero")
        model_service.ensure_initialized()

        # Acquire thread-safe GPU Lock
        with model_service.gpu_lock:
            if model_service.device == "cuda":
                torch.cuda.reset_peak_memory_stats(0)

            # 1. Generate Depth Anything V2 Small Relative Depth Prior
            self._synchronize(model_service.device)
            t_dav0 = time.time()
            with torch.inference_mode():
                dav2_out = model_service.dav2_model.predict_relative_depth(rgb_uint8, target_size=(H, W))
                dav2_prior = dav2_out['raw_depth'] if isinstance(dav2_out, dict) else dav2_out
            self._synchronize(model_service.device)
            dav2_time = time.time() - t_dav0

            # 2. M2-FINAL sliding-window inference with the validated Hanning
            # boundary floor. Small axes are padded, never resized.
            pad_h = max(0, self.crop_size - H)
            pad_w = max(0, self.crop_size - W)
            padded_h = H + pad_h
            padded_w = W + pad_w

            rgb_t = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).float()
            depth_t = torch.from_numpy(dav2_prior).unsqueeze(0).unsqueeze(0).float()
            if pad_h or pad_w:
                # Replication is defined for every input dimension, including 1px.
                rgb_t = F.pad(rgb_t, (0, pad_w, 0, pad_h), mode="replicate")
                depth_t = F.pad(depth_t, (0, pad_w, 0, pad_h), mode="replicate")

            positions_y = self._window_positions(padded_h)
            positions_x = self._window_positions(padded_w)

            # Preserve the exact sealed 1024 protocol: [0, 256, 512] x 2 axes.
            if (H, W) == (1024, 1024):
                if positions_y != [0, 256, 512] or positions_x != [0, 256, 512]:
                    raise RuntimeError("Validated 1024x1024 M2 window protocol changed")

            self._synchronize(model_service.device)
            t_rdah0 = time.time()
            pred_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
            weight_canvas = np.zeros((padded_h, padded_w), dtype=np.float32)

            with torch.inference_mode():
                for y0 in positions_y:
                    for x0 in positions_x:
                        y1 = y0 + self.crop_size
                        x1 = x0 + self.crop_size

                        rgb_patch = rgb_t[:, :, y0:y1, x0:x1].to(model_service.device)
                        depth_patch = depth_t[:, :, y0:y1, x0:x1].to(model_service.device)

                        patch_out = model_service.predict_patch(depth_patch, rgb_patch, gsd_m=gsd_m)
                        p_np = patch_out.squeeze().cpu().numpy().astype(np.float32)

                        pred_canvas[y0:y1, x0:x1] += p_np * self.window
                        weight_canvas[y0:y1, x0:x1] += self.window

            weight_canvas = np.maximum(weight_canvas, 1e-6)
            predicted_agl = (pred_canvas / weight_canvas)[:H, :W]
            # softplus/relu; this guard catches numerical/model-contract failure.
            if not np.isfinite(predicted_agl).all() or np.any(predicted_agl < 0):
                raise RuntimeError(f"{model_service.model_family} produced invalid non-finite or negative AGL")
            self._synchronize(model_service.device)
            rdah_time = time.time() - t_rdah0
            peak_vram_mb = (
                torch.cuda.max_memory_allocated(0) / (1024 * 1024)
                if model_service.device == "cuda"
                else 0.0
            )

        # Monotonically normalized scale-agnostic relative surface in [0, 1]
        # Larger value = higher elevation (closer to nadir overhead sensor).
        d_min = float(np.min(dav2_prior))
        d_max = float(np.max(dav2_prior))
        d_range = max(d_max - d_min, 1e-6)
        relative_surface = ((dav2_prior - d_min) / d_range).astype(np.float32)

        total_time = time.time() - t0
        return {
            'predicted_agl_m': predicted_agl,
            'relative_surface': relative_surface,
            'dav2_prior': dav2_prior,
            'dav2_time_s': dav2_time,
            'm2_time_s': rdah_time,
            'rdah_time_s': rdah_time,
            'ai_time_s': dav2_time + rdah_time,
            'total_inference_time_s': total_time,
            'peak_vram_mb': peak_vram_mb,
            'window_count': len(positions_y) * len(positions_x),
            'window_size_px': self.crop_size,
            'window_step_px': self.step_size,
            'blend': 'Hanning weighted with 0.05 boundary floor'
        }

inference_service = InferenceService()
