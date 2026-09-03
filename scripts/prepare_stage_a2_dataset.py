"""
DepthWizard (SIH26175) — Stage A2 Dataset Preparation & Downloader
Player 1: AI/ML Lead

Prepares expanded deterministic training split for Stage A2:
- Expands PHL/DC training dataset (from 1,000 to 2,000 unique tiles: 1,000 PHL + 1,000 DC)
- Keeps the exact 200 validation tiles (100 PHL + 100 DC) unchanged
- Keeps the locked 496 NYC test tiles unchanged
- Strict zero-leakage guarantee
"""

import os
import sys
import json
import random
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed

def prepare_stage_a2_splits(seed=42):
    print("=" * 80)
    print("STAGE A2 DATASET PREPARATION (SEED = 42)")
    print("=" * 80)

    random.seed(seed)
    out_dir = "data/gamus/splits"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load existing Validation Manifest (MUST REMAIN EXACTLY IDENTICAL)
    val_path = os.path.join(out_dir, "stage_a1_val.txt")
    with open(val_path, 'r', encoding='utf-8') as f:
        val_lines = [l.strip().split() for l in f if l.strip()]
    val_sids = set([p[0] for p in val_lines])
    print(f"Locked Validation Tiles: {len(val_sids)} (100 PHL + 100 DC)")

    # 2. Load locked NYC test manifest
    test_path = os.path.join(out_dir, "nyc_unseen_500_test.txt")
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [l.strip().split() for l in f if l.strip()]
    test_sids = set([p[0] for p in test_lines])
    print(f"Locked NYC Test Tiles: {len(test_sids)}")

    # 3. Query all available PHL and DC triplets from repository
    api = HfApi()
    all_files = set(api.list_repo_files('earthflow/GAMUS', repo_type='dataset'))
    img_files = [f for f in all_files if f.startswith('images/') and f.endswith('_RGB.h5')]

    available_triplets = []
    for f in img_files:
        parts = f.split('/')
        split = parts[1]
        fname = parts[2]
        sid = fname.replace('_RGB.h5', '')
        city = sid.split('_')[0]
        if city not in ['PHL', 'DC']:
            continue
        h_path = f"heights/{split}/{sid}_AGL.h5"
        c_path = f"classes/{split}/{sid}_CLS.h5"
        if h_path in all_files and c_path in all_files:
            # Check disjointness from val and test
            if sid not in val_sids and sid not in test_sids:
                available_triplets.append({
                    'sample_id': sid,
                    'hf_split': split,
                    'city': city,
                    'rgb_relpath': f,
                    'height_relpath': h_path,
                    'class_relpath': c_path
                })

    phl_avail = [t for t in available_triplets if t['city'] == 'PHL']
    dc_avail = [t for t in available_triplets if t['city'] == 'DC']
    print(f"Total candidate training triplets available (excluding val):")
    print(f"  Philadelphia (PHL): {len(phl_avail)}")
    print(f"  Washington DC (DC): {len(dc_avail)}")
    print(f"  Total Available:    {len(available_triplets)}")

    random.shuffle(phl_avail)
    random.shuffle(dc_avail)

    # 4. Create Stage A2 Training Set: 1,000 PHL + 1,000 DC = 2,000 unique tiles (2x expansion)
    target_train_count_per_city = 1000
    a2_train = phl_avail[:target_train_count_per_city] + dc_avail[:target_train_count_per_city]
    random.shuffle(a2_train)

    a2_train_path = os.path.join(out_dir, "stage_a2_train.txt")
    with open(a2_train_path, 'w', encoding='utf-8') as f:
        for t in a2_train:
            f.write(f"{t['sample_id']}\t{t['hf_split']}\t{t['rgb_relpath']}\t{t['height_relpath']}\t{t['class_relpath']}\n")

    print(f"\nSaved Stage A2 Train Manifest ({len(a2_train)} samples: {target_train_count_per_city} PHL + {target_train_count_per_city} DC): {a2_train_path}")

    # 5. Verify Zero Leakage
    a2_train_sids = set([t['sample_id'] for t in a2_train])
    assert len(a2_train_sids.intersection(val_sids)) == 0, "Leakage with validation set detected!"
    assert len(a2_train_sids.intersection(test_sids)) == 0, "Leakage with NYC test set detected!"
    print(">> DATASET LEAKAGE VERIFICATION: 100% DISJOINT (ZERO LEAKAGE PASS)")

    # 6. Download any missing files for the 2,000-tile dataset
    print(f"\n--- Downloading Missing Stage A2 Files ---")
    local_dir = "data/gamus/dataset"
    files_to_check = []
    for t in a2_train:
        files_to_check.extend([t['rgb_relpath'], t['height_relpath'], t['class_relpath']])

    missing_files = [f for f in files_to_check if not os.path.exists(os.path.join(local_dir, f))]
    print(f"Total files in Stage A2 train: {len(files_to_check)} ({len(a2_train)} triplets)")
    print(f"Files already locally cached: {len(files_to_check) - len(missing_files)}")
    print(f"Files to download:            {len(missing_files)}")

    if missing_files:
        def download_fn(fname):
            try:
                hf_hub_download(repo_id='earthflow/GAMUS', filename=fname, repo_type='dataset', local_dir=local_dir)
                return fname, True
            except Exception as e:
                return fname, False

        print("Starting multi-threaded download (16 threads)...")
        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(download_fn, fn): fn for fn in missing_files}
            for idx, fut in enumerate(as_completed(futures), 1):
                fname, ok = fut.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                if idx % 100 == 0 or idx == len(missing_files):
                    print(f"  Progress: {idx}/{len(missing_files)} ({success} downloaded, {failed} errors)")

        print(f"Download complete: {success} downloaded, {failed} errors.")

    return a2_train_path

if __name__ == '__main__':
    prepare_stage_a2_splits()
