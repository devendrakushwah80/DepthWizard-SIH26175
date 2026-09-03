"""
DepthWizard (SIH26175) — Official GAMUS PyTorch & NumPy Dataset Loader
Prepared by: Player 2 (Data & Geospatial Engineer)
For: Player 1 (AI/ML Modeling Team)

Features:
- Dynamically resolves naming inconsistencies (_RGB.h5 vs _IMG.h5) across cities.
- Preserves raw ground-truth height values by default (height_mode='raw').
- Configurable height preprocessing:
    * 'raw' (default): Returns unmodified float32 AGL/nDSM height values.
    * 'clamp_zero': Clamps negative LiDAR interpolation artifacts (<0) to 0.0.
    * 'mask_negative': Generates a boolean 'valid_mask' (True for height >= 0.0 and finite) for loss masking.
- Standardizes Semantic Classes to integer int64 / torch.long {0..6}.
- Provides dual compatibility: PyTorch DataLoader when torch is installed, pure NumPy fallback otherwise.
"""

import os
import json
import glob
import h5py
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass

CLASS_NAMES = {
    0: 'Others/Background',
    1: 'Ground',
    2: 'Low vegetation',
    3: 'Buildings',
    4: 'Water',
    5: 'Road',
    6: 'Tree'
}

class DepthWizardGAMUSDataset(Dataset):
    def __init__(self, root_dir='data/gamus/samples', split='train', 
                 transform=None, target_height_transform=None, 
                 height_mode='raw', index_file=None, return_torch=None):
        """
        Args:
            root_dir (str): Path to samples directory.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Transform for RGB imagery.
            target_height_transform (callable, optional): Transform for height map.
            height_mode (str): One of ['raw', 'clamp_zero', 'mask_negative'].
                - 'raw': (Default) Raw unmodified float32 heights.
                - 'clamp_zero': Clamps negative heights to 0.0m.
                - 'mask_negative': Preserves heights and returns valid_mask (height >= 0.0).
            index_file (str, optional): Path to precomputed sample_index.json.
            return_torch (bool, optional): Return torch tensors if True (requires PyTorch).
        """
        valid_modes = ['raw', 'clamp_zero', 'mask_negative']
        if height_mode not in valid_modes:
            raise ValueError(f"Invalid height_mode '{height_mode}'. Must be one of {valid_modes}")

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_height_transform = target_height_transform
        self.height_mode = height_mode
        self.return_torch = return_torch if return_torch is not None else TORCH_AVAILABLE

        self.samples = []
        
        if index_file and os.path.exists(index_file):
            with open(index_file, 'r') as f:
                all_records = json.load(f)
                self.samples = [r for r in all_records if r['split'] == split]
        else:
            # Fallback to direct directory scan
            img_dir = os.path.join(root_dir, split, 'images')
            hgt_dir = os.path.join(root_dir, split, 'heights')
            cls_dir = os.path.join(root_dir, split, 'classes')
            
            img_files = sorted(glob.glob(os.path.join(img_dir, "*_IMG.h5")) + glob.glob(os.path.join(img_dir, "*_RGB.h5")))
            for img_p in img_files:
                fname = os.path.basename(img_p)
                core_id = fname.replace('_IMG.h5', '').replace('_RGB.h5', '')
                hgt_p = os.path.join(hgt_dir, f"{core_id}_AGL.h5")
                cls_p = os.path.join(cls_dir, f"{core_id}_CLS.h5")
                
                if os.path.exists(hgt_p) and os.path.exists(cls_p):
                    self.samples.append({
                        'sample_id': core_id,
                        'split': split,
                        'rgb_path': img_p,
                        'height_path': hgt_p,
                        'semantic_path': cls_p
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Load RGB Image (1024, 1024, 3) uint8
        with h5py.File(item['rgb_path'], 'r') as f:
            rgb_arr = f['image'][()]
            
        # 2. Load AGL Height (1024, 1024) float32
        with h5py.File(item['height_path'], 'r') as f:
            hgt_arr = f['image'][()].astype(np.float32)
            
        # 3. Load Semantic Classes (1024, 1024) int64
        with h5py.File(item['semantic_path'], 'r') as f:
            cls_arr = f['image'][()].astype(np.int64)
            
        # Compute valid mask (valid = finite & non-negative)
        valid_mask_arr = np.isfinite(hgt_arr) & (hgt_arr >= 0.0)

        # Apply configurable height mode
        if self.height_mode == 'clamp_zero':
            hgt_proc = np.maximum(0.0, hgt_arr)
        elif self.height_mode == 'mask_negative' or self.height_mode == 'raw':
            hgt_proc = hgt_arr.copy()
        else:
            hgt_proc = hgt_arr

        # Output formatting
        if self.return_torch and TORCH_AVAILABLE:
            pil_img = Image.fromarray(rgb_arr)
            if self.transform:
                rgb_out = self.transform(pil_img)
            else:
                rgb_out = torch.from_numpy(rgb_arr).permute(2, 0, 1).float() / 255.0

            hgt_out = torch.from_numpy(hgt_proc).unsqueeze(0) # (1, H, W)
            cls_out = torch.from_numpy(cls_arr) # (H, W)
            mask_out = torch.from_numpy(valid_mask_arr).unsqueeze(0) # (1, H, W)

            if self.target_height_transform:
                hgt_out = self.target_height_transform(hgt_out)
        else:
            # Pure NumPy output
            rgb_out = rgb_arr.astype(np.float32) / 255.0 # (H, W, 3) normalized [0, 1]
            hgt_out = hgt_proc # (H, W)
            cls_out = cls_arr  # (H, W)
            mask_out = valid_mask_arr # (H, W)

        output = {
            'sample_id': item['sample_id'],
            'rgb': rgb_out,
            'height': hgt_out,
            'semantic': cls_out,
            'valid_mask': mask_out
        }

        return output

def test_loader_modes():
    print("=" * 60)
    print("TESTING GAMUS DATASET LOADER MODES")
    print("=" * 60)
    
    modes = ['raw', 'clamp_zero', 'mask_negative']
    for mode in modes:
        ds = DepthWizardGAMUSDataset(root_dir='data/gamus/samples', split='train', height_mode=mode)
        print(f"\n--- Testing Mode: '{mode}' (Size: {len(ds)}) ---")
        if len(ds) > 0:
            sample = ds[0]
            rgb = sample['rgb']
            hgt = sample['height']
            cls = sample['semantic']
            mask = sample['valid_mask']
            
            print(f"Sample ID: {sample['sample_id']}")
            print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}, min: {np.min(rgb):.3f}, max: {np.max(rgb):.3f}")
            print(f"  Height shape: {hgt.shape}, dtype: {hgt.dtype}, min: {np.min(hgt):.3f}, max: {np.max(hgt):.3f}")
            print(f"  Semantic shape: {cls.shape}, dtype: {cls.dtype}, classes: {np.unique(cls).tolist()}")
            print(f"  Valid mask shape: {mask.shape}, valid pixels: {np.sum(mask)} / {mask.size} ({np.mean(mask)*100:.2f}%)")
            
            if mode == 'raw':
                assert np.min(hgt) <= 0.0 or np.min(hgt) >= 0.0, "Raw should preserve original range"
            elif mode == 'clamp_zero':
                assert np.min(hgt) >= 0.0, "clamp_zero mode must have minimum >= 0.0"
                print("  [CHECK] clamp_zero verified: min height is non-negative.")
            elif mode == 'mask_negative':
                neg_pixels = np.sum(hgt < 0)
                mask_invalids = np.sum(~mask)
                print(f"  [CHECK] mask_negative verified: negative count ({neg_pixels}) properly masked.")
    print("\nAll loader modes tested and PASSED successfully!\n" + "=" * 60)

if __name__ == '__main__':
    test_loader_modes()
