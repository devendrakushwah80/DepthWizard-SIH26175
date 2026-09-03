"""
DepthWizard (SIH26175) — Depth Anything V2 Small Wrapper
Player 1: AI/ML Lead
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnythingV2Wrapper:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device="cpu"):
        """
        Initializes Depth Anything V2 Small model.
        Args:
            model_id (str): Hugging Face model repository or local checkpoint.
            device (str): 'cuda', 'cpu', etc.
        """
        self.model_id = model_id
        self.device = device
        
        t0 = time.time()
        # Prefer the already cached sealed dependency for deterministic/offline
        # product startup. Only contact Hugging Face when the cache is absent.
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id, local_files_only=True
            )
        except OSError:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.load_time = time.time() - t0
        
        self.total_params = sum(p.numel() for p in self.model.parameters())
        
    def predict_relative_depth(self, rgb_input, target_size=(1024, 1024), input_size=(518, 518)):
        """
        Infers scale-agnostic relative depth for an RGB image.
        Args:
            rgb_input (np.ndarray or PIL.Image or torch.Tensor): 
                - np.ndarray of shape (H, W, 3) in [0, 255] or [0, 1]
                - PIL Image
            target_size (tuple): Output grid size (H, W), default (1024, 1024).
            input_size (tuple): Inference resolution multiple of 14, default (518, 518).
        Returns:
            dict: {
                'raw_depth': np.ndarray of shape (target_size[0], target_size[1]),
                'inference_time_ms': float,
                'input_size': input_size,
                'target_size': target_size
            }
        """
        if isinstance(rgb_input, torch.Tensor):
            if rgb_input.ndim == 3 and rgb_input.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                rgb_input = rgb_input.permute(1, 2, 0).cpu().numpy()
            if rgb_input.max() <= 1.0:
                rgb_input = (rgb_input * 255.0).astype(np.uint8)
            else:
                rgb_input = rgb_input.astype(np.uint8)
                
        if isinstance(rgb_input, np.ndarray):
            if rgb_input.dtype != np.uint8:
                if rgb_input.max() <= 1.0:
                    rgb_input = (rgb_input * 255.0).astype(np.uint8)
                else:
                    rgb_input = np.clip(rgb_input, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(rgb_input)
        elif isinstance(rgb_input, Image.Image):
            pil_img = rgb_input
        else:
            raise ValueError(f"Unsupported RGB input type: {type(rgb_input)}")
            
        # Process input for model
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_depth = outputs.predicted_depth # Shape: (1, H_infer, W_infer)
            
            # Bilinear upsample back to target_size (e.g. 1024x1024)
            # pred_depth shape: (1, 1, H, W) for interpolate
            pred_depth_4d = pred_depth.unsqueeze(1)
            upsampled_depth = F.interpolate(
                pred_depth_4d,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
            raw_depth = upsampled_depth.squeeze().cpu().numpy().astype(np.float32)
            
        infer_time_ms = (time.time() - t0) * 1000.0
        
        return {
            'raw_depth': raw_depth,
            'inference_time_ms': infer_time_ms,
            'input_size': input_size,
            'target_size': target_size
        }
