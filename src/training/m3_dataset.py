"""
DepthWizard (SIH26175) — M3 Multi-Domain Dataset Pipeline
Author: DepthWizard Phase 2 Pipeline

Supports multi-domain inputs:
- GAMUS HDF5 (DC, PHL, NYC)
- US3D GeoTIFF (JAX, OMA)
- Natural Domain GeoTIFF (NAIP + 3DEP)
- Cached or on-the-fly DAV2 priors
- Height-aware crop sampling
- Geometric augmentations (D4 symmetry: rotations + flips)
- GSD conditioning inputs with GSD dropout
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import h5py
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class M3MultiDomainDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split_records_path: Optional[str | Path] = None,
        crop_size: int = 512,
        is_training: bool = True,
        height_aware_sampling: bool = False,
        gsd_dropout_prob: float = 0.0,
        cache_dav2_dir: Optional[str | Path] = "data/gamus/cache_dav2",
        dav2_model: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.crop_size = crop_size
        self.is_training = is_training
        self.height_aware_sampling = height_aware_sampling
        self.gsd_dropout_prob = gsd_dropout_prob
        self.cache_dav2_dir = Path(cache_dav2_dir) if cache_dav2_dir else None
        self.dav2_model = dav2_model
        self.device = device

        # Load records from manifest
        all_records_map = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    all_records_map[r["record_id"]] = r

        if split_records_path:
            with open(split_records_path, "r", encoding="utf-8") as f:
                split_ids = [line.strip() for line in f if line.strip()]
            self.records = [all_records_map[sid] for sid in split_ids if sid in all_records_map]
        else:
            self.records = list(all_records_map.values())

        print(f"M3MultiDomainDataset: Loaded {len(self.records)} records (is_training={is_training}, height_aware={height_aware_sampling})")

    def __len__(self) -> int:
        return len(self.records)

    def _load_gamus(self, rec: dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Load RGB
        with h5py.File(rec["rgb_path"], "r") as f:
            k = list(f.keys())[0]
            rgb = f[k][()]  # (1024, 1024, 3) uint8

        # Load AGL
        with h5py.File(rec["agl_path"], "r") as f:
            k = list(f.keys())[0]
            agl = f[k][()]  # (1024, 1024) float32

        # Load Semantics if present
        sem = None
        if rec.get("semantic_path") and Path(rec["semantic_path"]).exists():
            try:
                with h5py.File(rec["semantic_path"], "r") as f:
                    k = list(f.keys())[0]
                    sem = f[k][()]  # (1024, 1024) uint8
            except Exception:
                sem = None

        return rgb, agl, sem

    def _load_us3d(self, rec: dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Load optical GeoTIFF
        with rasterio.open(rec["rgb_path"]) as src:
            rgb = src.read()  # (3, H, W) uint8
            rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)

        # Load nDSM GeoTIFF
        with rasterio.open(rec["agl_path"]) as src:
            agl = src.read(1)  # (H, W) float32

        return rgb, agl, None

    def _load_natural(self, rec: dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Load NAIP GeoTIFF
        with rasterio.open(rec["rgb_path"]) as src:
            rgb = src.read([1, 2, 3])  # First 3 bands RGB (3, H, W)
            rgb = np.transpose(rgb, (1, 2, 0))

        # Bare terrain object height reference is ~0m
        with rasterio.open(rec["dem_path"]) as src:
            dem = src.read(1)
        agl = np.zeros_like(dem, dtype=np.float32)

        return rgb, agl, None

    def _get_dav2_prior(self, sid: str, rgb: np.ndarray) -> np.ndarray:
        # Check cache with multiple possible naming conventions
        if self.cache_dav2_dir:
            candidates = [
                self.cache_dav2_dir / f"{sid}_dav2.npy",
                self.cache_dav2_dir / f"{sid}_IMG.h5_dav2.npy",
                self.cache_dav2_dir / f"{sid}_RGB.h5_dav2.npy",
                self.cache_dav2_dir / f"{sid}.h5_dav2.npy",
            ]
            for cache_file in candidates:
                if cache_file.exists():
                    try:
                        prior = np.load(cache_file)
                        if prior.shape == rgb.shape[:2]:
                            return prior
                    except Exception:
                        pass

        # Extract on-the-fly if model provided
        if self.dav2_model is not None:
            with torch.inference_mode():
                # Prepare tensor
                t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                t_rgb = t_rgb.to(self.device)
                prior_tensor = self.dav2_model(t_rgb)
                prior = prior_tensor.squeeze().cpu().numpy()
                # Normalize to [0, 1]
                p_min, p_max = prior.min(), prior.max()
                if p_max > p_min:
                    prior = (prior - p_min) / (p_max - p_min)
                else:
                    prior = np.zeros_like(prior)
                return prior.astype(np.float32)

        # Fallback pseudo-prior based on image intensity
        gray = np.mean(rgb.astype(np.float32) / 255.0, axis=-1)
        return gray.astype(np.float32)

    def _sample_crop(self, rgb: np.ndarray, agl: np.ndarray, prior: np.ndarray, sem: Optional[np.ndarray]):
        h, w = rgb.shape[:2]
        if h <= self.crop_size and w <= self.crop_size:
            return rgb, agl, prior, sem

        max_y = h - self.crop_size
        max_x = w - self.crop_size

        if self.is_training and self.height_aware_sampling and random.random() < 0.65:
            # 65% chance to sample an anchored crop:
            # 35% tall structure (>=10m) + 30% canopy/vegetation (4m - 25m)
            stride = 16
            sub_agl = agl[::stride, ::stride]
            if random.random() < 0.50:
                anchors_y, anchors_x = np.where(sub_agl >= 10.0)
            else:
                anchors_y, anchors_x = np.where((sub_agl >= 4.0) & (sub_agl <= 25.0))

            if len(anchors_y) == 0:
                anchors_y, anchors_x = np.where(sub_agl >= 4.0)

            if len(anchors_y) > 0:
                idx = random.randint(0, len(anchors_y) - 1)
                cy = anchors_y[idx] * stride
                cx = anchors_x[idx] * stride
                top = max(0, min(max_y, cy - self.crop_size // 2 + random.randint(-64, 64)))
                left = max(0, min(max_x, cx - self.crop_size // 2 + random.randint(-64, 64)))
            else:
                top = random.randint(0, max_y)
                left = random.randint(0, max_x)
        elif self.is_training:
            top = random.randint(0, max_y)
            left = random.randint(0, max_x)
        else:
            # Deterministic center crop for validation
            top = max_y // 2
            left = max_x // 2

        rgb_c = rgb[top:top + self.crop_size, left:left + self.crop_size]
        agl_c = agl[top:top + self.crop_size, left:left + self.crop_size]
        prior_c = prior[top:top + self.crop_size, left:left + self.crop_size]
        sem_c = sem[top:top + self.crop_size, left:left + self.crop_size] if sem is not None else None

        return rgb_c, agl_c, prior_c, sem_c

    def _apply_augmentation(
        self, rgb: np.ndarray, agl: np.ndarray, prior: np.ndarray, sem: Optional[np.ndarray]
    ):
        # 1. Random 90-degree rotations (0, 1, 2, or 3 times 90 deg)
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            rgb = np.ascontiguousarray(np.rot90(rgb, k=k_rot, axes=(0, 1)))
            agl = np.ascontiguousarray(np.rot90(agl, k=k_rot, axes=(0, 1)))
            prior = np.ascontiguousarray(np.rot90(prior, k=k_rot, axes=(0, 1)))
            if sem is not None:
                sem = np.ascontiguousarray(np.rot90(sem, k=k_rot, axes=(0, 1)))

        # 2. Random horizontal flip
        if random.random() > 0.5:
            rgb = np.ascontiguousarray(np.fliplr(rgb))
            agl = np.ascontiguousarray(np.fliplr(agl))
            prior = np.ascontiguousarray(np.fliplr(prior))
            if sem is not None:
                sem = np.ascontiguousarray(np.fliplr(sem))

        # 3. Random vertical flip
        if random.random() > 0.5:
            rgb = np.ascontiguousarray(np.flipud(rgb))
            agl = np.ascontiguousarray(np.flipud(agl))
            prior = np.ascontiguousarray(np.flipud(prior))
            if sem is not None:
                sem = np.ascontiguousarray(np.flipud(sem))

        # 4. Moderate color jitter (RGB only)
        if random.random() > 0.5:
            factor_b = random.uniform(0.85, 1.15)
            factor_c = random.uniform(0.85, 1.15)
            rgb_f = rgb.astype(np.float32) * factor_b
            mean = rgb_f.mean(axis=(0, 1), keepdims=True)
            rgb_f = (rgb_f - mean) * factor_c + mean
            rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

        return rgb, agl, prior, sem

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        dataset = rec["dataset"]
        sid = rec["record_id"].replace("GAMUS_", "").replace("US3D_", "").replace("NATURAL_", "")

        if "GAMUS" in dataset:
            rgb, agl, sem = self._load_gamus(rec)
        elif "US3D" in dataset:
            rgb, agl, sem = self._load_us3d(rec)
        elif "NATURAL" in dataset:
            rgb, agl, sem = self._load_natural(rec)
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")

        # Ensure valid float32 heights and NaN masking
        agl = agl.astype(np.float32)
        valid_mask = np.isfinite(agl) & (agl >= 0.0)
        agl[~valid_mask] = 0.0

        # Get DAV2 relative prior
        prior = self._get_dav2_prior(sid, rgb)

        # Crop to crop_size
        rgb_c, agl_c, prior_c, sem_c = self._sample_crop(rgb, agl, prior, sem)

        # Augmentation during training
        if self.is_training:
            rgb_c, agl_c, prior_c, sem_c = self._apply_augmentation(rgb_c, agl_c, prior_c, sem_c)

        # Recompute mask on cropped patch
        valid_mask_c = np.isfinite(agl_c) & (agl_c >= 0.0)

        # Convert to PyTorch Tensors
        # RGB normalized to [0, 1]
        t_rgb = torch.from_numpy(rgb_c).permute(2, 0, 1).float() / 255.0
        # Normalization with ImageNet mean/std
        t_rgb = TF.normalize(t_rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Prior (1, H, W)
        t_prior = torch.from_numpy(prior_c).unsqueeze(0).float()
        # Metric AGL target (1, H, W)
        t_agl = torch.from_numpy(agl_c).unsqueeze(0).float()
        # Valid mask (1, H, W)
        t_mask = torch.from_numpy(valid_mask_c).unsqueeze(0).float()

        # GSD conditioning
        gsd_m = float(rec.get("gsd_m", 0.5))
        gsd_known = bool(rec.get("gsd_known", True))

        # GSD dropout during training for robustness
        if self.is_training and self.gsd_dropout_prob > 0.0 and random.random() < self.gsd_dropout_prob:
            gsd_known = False
            gsd_m = 0.5  # default nominal value

        sample = {
            "record_id": rec["record_id"],
            "dataset": dataset,
            "city": rec["city"],
            "rgb": t_rgb,
            "prior": t_prior,
            "agl": t_agl,
            "mask": t_mask,
            "gsd_m": torch.tensor(gsd_m, dtype=torch.float32),
            "gsd_known": torch.tensor(1.0 if gsd_known else 0.0, dtype=torch.float32)
        }

        if sem_c is not None:
            sample["semantics"] = torch.from_numpy(sem_c).long()
        else:
            sample["semantics"] = torch.full((self.crop_size, self.crop_size), -1, dtype=torch.long)

        return sample
