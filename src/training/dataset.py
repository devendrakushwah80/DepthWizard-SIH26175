"""
DepthWizard (SIH26175) — GAMUS Stage A PyTorch Dataset & Augmentation Pipeline
Player 1: AI/ML Lead
"""

import os
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class GAMUSStageADataset(Dataset):
    def __init__(self, manifest_path, data_dir="data/gamus/dataset", dav2_cache_dir=None,
                 crop_size=(512, 512), is_training=True, transform_geo=True):
        """
        Args:
            manifest_path (str): Path to manifest text file (e.g. data/gamus/splits/stage_a0_train.txt)
            data_dir (str): Root directory where downloaded H5 files reside
            dav2_cache_dir (str, optional): Directory containing cached DAV2 relative depth .npy or .h5 files
            crop_size (tuple): (H, W) crop dimensions, default (512, 512)
            is_training (bool): If True, applies random cropping and geometry-preserving augmentations
            transform_geo (bool): If True, applies random flips and 90-degree rotations
        """
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.dav2_cache_dir = dav2_cache_dir
        self.crop_size = crop_size
        self.is_training = is_training
        self.transform_geo = transform_geo

        self.samples = []
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        sid, hf_split, rgb_rel, hgt_rel, cls_rel = parts[0], parts[1], parts[2], parts[3], parts[4]
                        
                        rgb_full = os.path.join(data_dir, rgb_rel)
                        hgt_full = os.path.join(data_dir, hgt_rel)
                        cls_full = os.path.join(data_dir, cls_rel)
                        
                        self.samples.append({
                            'sample_id': sid,
                            'split': hf_split,
                            'rgb_path': rgb_full,
                            'height_path': hgt_full,
                            'class_path': cls_full
                        })

    def __len__(self):
        return len(self.samples)

    def _read_h5(self, path):
        with h5py.File(path, 'r') as h5f:
            keys = list(h5f.keys())
            if not keys:
                raise ValueError(f"Empty H5 file: {path}")
            data = np.array(h5f[keys[0]])
        return data

    def __getitem__(self, idx):
        item = self.samples[idx]
        sid = item['sample_id']

        # 1. Load raw H5 arrays
        rgb_raw = self._read_h5(item['rgb_path']) # (1024, 1024, 3) or (3, 1024, 1024)
        hgt_raw = self._read_h5(item['height_path']) # (1024, 1024)
        cls_raw = self._read_h5(item['class_path']) # (1024, 1024)

        # Standardize shapes to (H, W, C)
        if rgb_raw.ndim == 3 and rgb_raw.shape[0] == 3:
            rgb_raw = np.transpose(rgb_raw, (1, 2, 0))

        # Standardize types and ranges
        if rgb_raw.dtype == np.uint8 or rgb_raw.max() > 1.0:
            rgb = rgb_raw.astype(np.float32) / 255.0
        else:
            rgb = rgb_raw.astype(np.float32)

        hgt = hgt_raw.astype(np.float32)
        cls = cls_raw.astype(np.int64)

        # Create valid mask
        valid_mask = (hgt >= 0.0) & np.isfinite(hgt) & (cls >= 0) & (cls <= 6)

        # 2. Check for cached DAV2 depth
        rel_depth = None
        if self.dav2_cache_dir is not None:
            cache_file = os.path.join(self.dav2_cache_dir, f"{sid}_dav2.npy")
            if os.path.exists(cache_file):
                rel_depth = np.load(cache_file).astype(np.float32)

        H, W = hgt.shape
        crop_h, crop_w = self.crop_size

        # 3. Crop Strategy
        if self.is_training:
            # Random Crop
            max_top = max(0, H - crop_h)
            max_left = max(0, W - crop_w)
            top = random.randint(0, max_top) if max_top > 0 else 0
            left = random.randint(0, max_left) if max_left > 0 else 0
        else:
            # Center Crop
            top = max(0, (H - crop_h) // 2)
            left = max(0, (W - crop_w) // 2)

        rgb_crop = rgb[top:top+crop_h, left:left+crop_w, :]
        hgt_crop = hgt[top:top+crop_h, left:left+crop_w]
        cls_crop = cls[top:top+crop_h, left:left+crop_w]
        mask_crop = valid_mask[top:top+crop_h, left:left+crop_w]
        
        if rel_depth is not None:
            depth_crop = rel_depth[top:top+crop_h, left:left+crop_w]
        else:
            depth_crop = np.zeros((crop_h, crop_w), dtype=np.float32)

        # 4. Data Augmentation (Geometry Preserving)
        if self.is_training and self.transform_geo:
            # Horizontal Flip
            if random.random() > 0.5:
                rgb_crop = np.fliplr(rgb_crop)
                hgt_crop = np.fliplr(hgt_crop)
                cls_crop = np.fliplr(cls_crop)
                mask_crop = np.fliplr(mask_crop)
                depth_crop = np.fliplr(depth_crop)

            # Vertical Flip
            if random.random() > 0.5:
                rgb_crop = np.flipud(rgb_crop)
                hgt_crop = np.flipud(hgt_crop)
                cls_crop = np.flipud(cls_crop)
                mask_crop = np.flipud(mask_crop)
                depth_crop = np.flipud(depth_crop)

            # Random 90-degree Rotation
            k_rot = random.choice([0, 1, 2, 3])
            if k_rot > 0:
                rgb_crop = np.rot90(rgb_crop, k=k_rot)
                hgt_crop = np.rot90(hgt_crop, k=k_rot)
                cls_crop = np.rot90(cls_crop, k=k_rot)
                mask_crop = np.rot90(mask_crop, k=k_rot)
                depth_crop = np.rot90(depth_crop, k=k_rot)

            # Subtle photometric jitter (brightness / contrast)
            if random.random() > 0.5:
                b_factor = random.uniform(0.9, 1.1)
                c_factor = random.uniform(0.9, 1.1)
                rgb_crop = np.clip((rgb_crop - 0.5) * c_factor + 0.5 + (b_factor - 1.0), 0.0, 1.0)

        # Convert to PyTorch Tensors
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_crop, (2, 0, 1)))).float() # (3, H, W)
        hgt_tensor = torch.from_numpy(np.ascontiguousarray(hgt_crop)).unsqueeze(0).float() # (1, H, W)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_crop)).unsqueeze(0).bool() # (1, H, W)
        cls_tensor = torch.from_numpy(np.ascontiguousarray(cls_crop)).long() # (H, W)
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_crop)).unsqueeze(0).float() # (1, H, W)

        return {
            'sample_id': sid,
            'rgb': rgb_tensor,
            'height': hgt_tensor,
            'valid_mask': mask_tensor,
            'semantic': cls_tensor,
            'rel_depth': depth_tensor,
            'has_cached_depth': (rel_depth is not None)
        }
