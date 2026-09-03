"""
DepthWizard (SIH26175) — Staged Dataset Manifest Generator
Player 1: AI/ML Lead

Prepares deterministic, reproducible splits for:
- Stage A0: Debug Pipeline (128 Train, 32 Val from PHL and DC)
- Stage A1: Real Initial Baseline (1,000 Train: 500 PHL + 500 DC; 200 Val: 100 PHL + 100 DC)
- Unseen Geographic Test: All NYC samples (New York City strictly withheld from training and validation)
"""

import os
import sys
import json
import random
from huggingface_hub import HfApi

def prepare_splits(seed=42):
    print("=" * 80)
    print("GENERATING DETERMINISTIC STAGE A SPLIT MANIFESTS (SEED = 42)")
    print("=" * 80)
    
    random.seed(seed)
    out_dir = "data/gamus/splits"
    os.makedirs(out_dir, exist_ok=True)
    
    api = HfApi()
    print("Fetching repository file list from earthflow/GAMUS...")
    all_files = api.list_repo_files('earthflow/GAMUS', repo_type='dataset')
    
    # Identify RGB image files
    img_files = [f for f in all_files if f.startswith('images/') and f.endswith('.h5')]
    print(f"Total image files in repository: {len(img_files)}")
    
    # Index all triplets
    triplets = []
    for f in img_files:
        # e.g. 'images/train/DC_28_45_RGB.h5'
        parts = f.split('/')
        hf_split = parts[1] # 'train', 'val', 'test'
        fname = parts[2]
        sid = fname.replace('_RGB.h5', '')
        city = sid.split('_')[0] # 'PHL', 'DC', 'NYC'
        
        hgt_path = f"heights/{hf_split}/{sid}_AGL.h5"
        cls_path = f"classes/{hf_split}/{sid}_CLS.h5"
        
        triplets.append({
            'sample_id': sid,
            'hf_split': hf_split,
            'city': city,
            'rgb_relpath': f,
            'height_relpath': hgt_path,
            'class_relpath': cls_path
        })

    print(f"Total matched triplets: {len(triplets)}")
    
    # Categorize by city
    phl_samples = [t for t in triplets if t['city'] == 'PHL']
    dc_samples = [t for t in triplets if t['city'] == 'DC']
    nyc_samples = [t for t in triplets if t['city'] == 'NYC']
    
    print(f"\nSamples by City:")
    print(f"  Philadelphia (PHL): {len(phl_samples):,}")
    print(f"  Washington D.C. (DC): {len(dc_samples):,}")
    print(f"  New York City (NYC): {len(nyc_samples):,} (Withheld for Unseen Out-of-Domain Evaluation)")
    
    # Shuffle deterministically
    random.shuffle(phl_samples)
    random.shuffle(dc_samples)
    random.shuffle(nyc_samples)
    
    # 1. NYC Unseen Test Manifest
    nyc_manifest_path = os.path.join(out_dir, "nyc_unseen_test.txt")
    with open(nyc_manifest_path, 'w', encoding='utf-8') as f:
        for t in sorted(nyc_samples, key=lambda x: x['sample_id']):
            f.write(f"{t['sample_id']}\t{t['hf_split']}\t{t['rgb_relpath']}\t{t['height_relpath']}\t{t['class_relpath']}\n")
    print(f"\nSaved NYC Unseen Test manifest ({len(nyc_samples)} samples): {nyc_manifest_path}")
    
    # 2. Stage A0: Debug Pipeline Subset (128 Train, 32 Val)
    # 64 PHL + 64 DC for train; 16 PHL + 16 DC for val
    a0_train = phl_samples[:64] + dc_samples[:64]
    a0_val = phl_samples[64:80] + dc_samples[64:80]
    random.shuffle(a0_train)
    random.shuffle(a0_val)
    
    a0_train_path = os.path.join(out_dir, "stage_a0_train.txt")
    with open(a0_train_path, 'w', encoding='utf-8') as f:
        for t in a0_train:
            f.write(f"{t['sample_id']}\t{t['hf_split']}\t{t['rgb_relpath']}\t{t['height_relpath']}\t{t['class_relpath']}\n")
            
    a0_val_path = os.path.join(out_dir, "stage_a0_val.txt")
    with open(a0_val_path, 'w', encoding='utf-8') as f:
        for t in a0_val:
            f.write(f"{t['sample_id']}\t{t['hf_split']}\t{t['rgb_relpath']}\t{t['height_relpath']}\t{t['class_relpath']}\n")
            
    print(f"Saved Stage A0 Train manifest ({len(a0_train)} samples): {a0_train_path}")
    print(f"Saved Stage A0 Val manifest   ({len(a0_val)} samples): {a0_val_path}")

    # 3. Stage A1: Real Initial Baseline (1,000 Train: 500 PHL + 500 DC; 200 Val: 100 PHL + 100 DC)
    # Allocate without overlap between train and val
    a1_train = phl_samples[:500] + dc_samples[:500]
    a1_val = phl_samples[500:600] + dc_samples[500:600]
    random.shuffle(a1_train)
    random.shuffle(a1_val)
    
    a1_train_path = os.path.join(out_dir, "stage_a1_train.txt")
    with open(a1_train_path, 'w', encoding='utf-8') as f:
        for t in a1_train:
            f.write(f"{t['sample_id']}\t{t['hf_split']}\t{t['rgb_relpath']}\t{t['height_relpath']}\t{t['class_relpath']}\n")
            
    a1_val_path = os.path.join(out_dir, "stage_a1_val.txt")
    with open(a1_val_path, 'w', encoding='utf-8') as f:
        for t in a1_val:
            f.write(f"{t['sample_id']}\t{t['hf_split']}\t{t['rgb_relpath']}\t{t['height_relpath']}\t{t['class_relpath']}\n")
            
    print(f"Saved Stage A1 Train manifest ({len(a1_train)} samples): {a1_train_path}")
    print(f"Saved Stage A1 Val manifest   ({len(a1_val)} samples): {a1_val_path}")

    # Save summary metadata JSON
    meta = {
        'seed': seed,
        'total_dataset_triplets': len(triplets),
        'phl_total': len(phl_samples),
        'dc_total': len(dc_samples),
        'nyc_total': len(nyc_samples),
        'stage_a0': {
            'train_count': len(a0_train),
            'val_count': len(a0_val),
            'train_phl': 64, 'train_dc': 64,
            'val_phl': 16, 'val_dc': 16
        },
        'stage_a1': {
            'train_count': len(a1_train),
            'val_count': len(a1_val),
            'train_phl': 500, 'train_dc': 500,
            'val_phl': 100, 'val_dc': 100
        },
        'nyc_unseen_test_count': len(nyc_samples)
    }
    
    meta_path = os.path.join(out_dir, "manifest_summary.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved manifest summary: {meta_path}")
    print("=" * 80)

if __name__ == '__main__':
    prepare_splits()
