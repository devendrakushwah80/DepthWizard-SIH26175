import os
import glob
import json
import csv
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Official GAMUS Semantic Classes
CLASS_NAMES = {
    0: 'Others/Background',
    1: 'Ground',
    2: 'Low vegetation',
    3: 'Buildings',
    4: 'Water',
    5: 'Road',
    6: 'Tree'
}

# Color palette for classes (RGB normalized 0-1)
CLASS_COLORS = {
    0: '#000000', # Black / Others
    1: '#808080', # Gray / Ground
    2: '#90EE90', # Light Green / Low Veg
    3: '#FF0000', # Red / Buildings
    4: '#0000FF', # Blue / Water
    5: '#FFFF00', # Yellow / Road
    6: '#006400', # Dark Green / Tree
}

def inspect_h5_file(filepath):
    """Deep inspection of a single HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        first_key = keys[0] if keys else None
        data = f[first_key][()] if first_key else None
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if data is None:
        return {'keys': keys, 'size_mb': file_size_mb, 'error': 'No dataset found'}
    
    # Statistical analysis
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())
    total_elements = int(data.size)
    
    # Valid mask
    valid_data = data[np.isfinite(data)]
    
    if len(valid_data) > 0:
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
        mean_val = float(np.mean(valid_data))
        median_val = float(np.median(valid_data))
        std_val = float(np.std(valid_data))
        p95_val = float(np.percentile(valid_data, 95))
        p99_val = float(np.percentile(valid_data, 99))
        neg_count = int((valid_data < 0).sum())
        zero_count = int((valid_data == 0).sum())
    else:
        min_val = max_val = mean_val = median_val = std_val = p95_val = p99_val = None
        neg_count = zero_count = 0
        
    return {
        'keys': keys,
        'shape': list(data.shape),
        'dtype': str(data.dtype),
        'size_mb': file_size_mb,
        'total_elements': total_elements,
        'nan_count': nan_count,
        'nan_pct': (nan_count / total_elements) * 100 if total_elements > 0 else 0,
        'inf_count': inf_count,
        'inf_pct': (inf_count / total_elements) * 100 if total_elements > 0 else 0,
        'neg_count': neg_count,
        'neg_pct': (neg_count / len(valid_data)) * 100 if len(valid_data) > 0 else 0,
        'zero_count': zero_count,
        'zero_pct': (zero_count / len(valid_data)) * 100 if len(valid_data) > 0 else 0,
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'p95': p95_val,
        'p99': p99_val,
    }

def run_full_audit():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE GAMUS DATASET AUDIT (PLAYER 2)")
    print("=" * 70)
    
    samples_dir = "data/gamus/samples"
    out_vis_dir = "outputs/data_audit"
    metadata_dir = "data/gamus/metadata"
    os.makedirs(out_vis_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    splits = ['train', 'val', 'test']
    triplets = []

    for split in splits:
        img_dir = os.path.join(samples_dir, split, 'images')
        hgt_dir = os.path.join(samples_dir, split, 'heights')
        cls_dir = os.path.join(samples_dir, split, 'classes')

        if not os.path.exists(img_dir):
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, "*_IMG.h5")) + glob.glob(os.path.join(img_dir, "*_RGB.h5")))
        for img_path in img_files:
            filename = os.path.basename(img_path)
            if filename.endswith('_IMG.h5'):
                core_id = filename.replace('_IMG.h5', '')
            elif filename.endswith('_RGB.h5'):
                core_id = filename.replace('_RGB.h5', '')
            else:
                core_id = filename.split('.')[0]
            city = core_id.split('_')[0]
            
            hgt_path = os.path.join(hgt_dir, f"{core_id}_AGL.h5")
            cls_path = os.path.join(cls_dir, f"{core_id}_CLS.h5")
            
            triplets.append({
                'sample_id': core_id,
                'split': split,
                'city': city,
                'img_path': img_path,
                'hgt_path': hgt_path,
                'cls_path': cls_path,
                'has_hgt': os.path.exists(hgt_path),
                'has_cls': os.path.exists(cls_path)
            })

    print(f"Found {len(triplets)} triplets to audit.")

    # Global Accumulators
    global_class_counts = {cid: 0 for cid in CLASS_NAMES}
    global_unknown_classes = {}
    class_height_values = {cid: [] for cid in CLASS_NAMES}
    all_valid_heights = []
    
    sample_index_records = []
    detailed_h5_audit = []
    pixel_alignment_results = []
    
    for idx, item in enumerate(triplets, 1):
        core_id = item['sample_id']
        split = item['split']
        city = item['city']
        print(f"\n[{idx}/{len(triplets)}] Auditing Sample: {core_id} ({split}, {city})")
        
        # Load arrays
        with h5py.File(item['img_path'], 'r') as f:
            img_arr = f['image'][()]
        with h5py.File(item['hgt_path'], 'r') as f:
            hgt_arr = f['image'][()]
        with h5py.File(item['cls_path'], 'r') as f:
            cls_arr = f['image'][()]

        # H5 inspections
        img_info = inspect_h5_file(item['img_path'])
        hgt_info = inspect_h5_file(item['hgt_path'])
        cls_info = inspect_h5_file(item['cls_path'])

        # Pixel Alignment Check
        # img shape is (H, W, C), hgt is (H, W), cls is (H, W)
        img_hw = img_arr.shape[:2]
        hgt_hw = hgt_arr.shape[:2]
        cls_hw = cls_arr.shape[:2]

        is_aligned = (img_hw == hgt_hw == cls_hw)
        alignment_status = "PASS" if is_aligned else "FAIL"
        
        pixel_alignment_results.append({
            'sample_id': core_id,
            'split': split,
            'city': city,
            'img_shape': str(img_arr.shape),
            'hgt_shape': str(hgt_arr.shape),
            'cls_shape': str(cls_arr.shape),
            'aligned': is_aligned,
            'status': alignment_status
        })

        # Semantic class analysis for this sample
        unique_classes, counts = np.unique(cls_arr, return_counts=True)
        sample_classes_present = []
        for u_cls, count in zip(unique_classes, counts):
            u_cls_int = int(u_cls)
            sample_classes_present.append(f"{u_cls_int}:{CLASS_NAMES.get(u_cls_int, 'Unknown')}")
            if u_cls_int in global_class_counts:
                global_class_counts[u_cls_int] += int(count)
            else:
                global_unknown_classes[u_cls_int] = global_unknown_classes.get(u_cls_int, 0) + int(count)

        # Per-class height extraction
        valid_hgt_mask = np.isfinite(hgt_arr)
        for cid in CLASS_NAMES:
            c_mask = (cls_arr == cid) & valid_hgt_mask
            c_heights = hgt_arr[c_mask]
            if len(c_heights) > 0:
                # Subsample up to 20,000 pixels per class per image for memory
                if len(c_heights) > 20000:
                    c_heights = np.random.choice(c_heights, 20000, replace=False)
                class_height_values[cid].extend(c_heights.tolist())

        # Collect overall valid heights
        valid_flat = hgt_arr[valid_hgt_mask].flatten()
        if len(valid_flat) > 50000:
            sub_valid = np.random.choice(valid_flat, 50000, replace=False)
            all_valid_heights.extend(sub_valid.tolist())
        else:
            all_valid_heights.extend(valid_flat.tolist())

        # Generate Visual Data Audit Figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        
        # 1. RGB
        # If float and in [0, 255] or uint8
        if img_arr.dtype == np.uint8:
            vis_img = img_arr
        elif img_arr.max() > 1.0:
            vis_img = np.clip(img_arr, 0, 255).astype(np.uint8)
        else:
            vis_img = np.clip(img_arr * 255, 0, 255).astype(np.uint8)
        
        axes[0].imshow(vis_img)
        axes[0].set_title(f"RGB Optical Image\nShape: {img_arr.shape}, Dtype: {img_arr.dtype}", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # 2. Height / AGL Heatmap
        hgt_disp = np.copy(hgt_arr)
        # Use viridis/turbo colormap with sensible vmin/vmax
        h_min, h_max = np.nanmin(hgt_disp), np.nanmax(hgt_disp)
        im_h = axes[1].imshow(hgt_disp, cmap='turbo', vmin=0, vmax=max(30.0, np.percentile(valid_flat, 98) if len(valid_flat)>0 else 30))
        cbar = fig.colorbar(im_h, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Height / AGL Value (unscaled)', fontsize=9)
        axes[1].set_title(f"AGL / nDSM Height Map\nMin: {h_min:.2f}, Max: {h_max:.2f}, Mean: {np.nanmean(hgt_disp):.2f}", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # 3. Semantic Mask
        # Build discrete colormap for classes 0 to 6
        cmap_colors = [CLASS_COLORS[i] for i in range(7)]
        cmap = ListedColormap(cmap_colors)
        norm = BoundaryNorm(range(8), cmap.N)
        
        axes[2].imshow(cls_arr, cmap=cmap, norm=norm, interpolation='nearest')
        axes[2].set_title(f"Semantic Segmentation Mask\nClasses Present: {len(unique_classes)}/7", fontsize=11, fontweight='bold')
        axes[2].axis('off')
        
        # Legend for classes present
        patches = [mpatches.Patch(color=CLASS_COLORS[int(c)], label=f"{int(c)}: {CLASS_NAMES.get(int(c), 'Other')}") for c in unique_classes if int(c) in CLASS_COLORS]
        axes[2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True)

        fig.suptitle(f"GAMUS Data Audit | Sample: {core_id} | Split: {split.upper()} | City: {city} | Alignment: {alignment_status}", fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        vis_save_path = os.path.join(out_vis_dir, f"{split}_{core_id}_audit.png")
        plt.savefig(vis_save_path, bbox_inches='tight')
        plt.close(fig)

        # Record for Index
        sample_index_records.append({
            'sample_id': core_id,
            'split': split,
            'city': city,
            'rgb_path': item['img_path'],
            'height_path': item['hgt_path'],
            'semantic_path': item['cls_path'],
            'image_height': img_arr.shape[0],
            'image_width': img_arr.shape[1],
            'image_channels': img_arr.shape[2] if len(img_arr.shape) > 2 else 1,
            'rgb_dtype': str(img_arr.dtype),
            'rgb_min': float(np.min(img_arr)),
            'rgb_max': float(np.max(img_arr)),
            'height_min': hgt_info['min'],
            'height_max': hgt_info['max'],
            'height_mean': hgt_info['mean'],
            'height_median': hgt_info['median'],
            'height_std': hgt_info['std'],
            'height_p95': hgt_info['p95'],
            'height_p99': hgt_info['p99'],
            'height_nan_count': hgt_info['nan_count'],
            'height_neg_count': hgt_info['neg_count'],
            'height_zero_count': hgt_info['zero_count'],
            'classes_present': sample_classes_present,
            'alignment_status': alignment_status
        })
        
        detailed_h5_audit.append({
            'sample_id': core_id,
            'split': split,
            'city': city,
            'img_audit': img_info,
            'hgt_audit': hgt_info,
            'cls_audit': cls_info
        })

    # Save Machine-Readable Index
    index_json_path = os.path.join(metadata_dir, "sample_index.json")
    with open(index_json_path, 'w') as f:
        json.dump(sample_index_records, f, indent=2)
    print(f"\nSaved machine-readable JSON index: {index_json_path}")

    index_csv_path = os.path.join(metadata_dir, "sample_index.csv")
    if sample_index_records:
        keys = list(sample_index_records[0].keys())
        with open(index_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in sample_index_records:
                row_copy = dict(r)
                row_copy['classes_present'] = '; '.join(row_copy['classes_present'])
                writer.writerow(row_copy)
    print(f"Saved machine-readable CSV index: {index_csv_path}")

    # Generate Height Distribution Buckets & Histogram
    all_valid_heights = np.array(all_valid_heights)
    total_valid_pixels = len(all_valid_heights)
    
    buckets = {
        '0-2': int(np.sum((all_valid_heights >= 0) & (all_valid_heights <= 2))),
        '2-10': int(np.sum((all_valid_heights > 2) & (all_valid_heights <= 10))),
        '10-20': int(np.sum((all_valid_heights > 10) & (all_valid_heights <= 20))),
        '20-50': int(np.sum((all_valid_heights > 20) & (all_valid_heights <= 50))),
        '>50': int(np.sum(all_valid_heights > 50)),
        '<0 (Negative)': int(np.sum(all_valid_heights < 0))
    }
    
    bucket_pcts = {k: float((v / total_valid_pixels) * 100) for k, v in buckets.items()}
    
    # Save Height Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Bucket Bar Chart
    ax1.bar(bucket_pcts.keys(), bucket_pcts.values(), color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax1.set_title("Height Distribution by Value Buckets", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Height Value Ranges", fontsize=10)
    ax1.set_ylabel("Percentage of Pixels (%)", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for k, v in bucket_pcts.items():
        ax1.text(k, v + 1.0, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, max(bucket_pcts.values()) + 10)

    # Log/Histogram of continuous heights
    clipped_heights = np.clip(all_valid_heights, -5, 80)
    ax2.hist(clipped_heights, bins=50, color='#d95f02', edgecolor='black', alpha=0.75, density=True)
    ax2.set_title("Height Value Density Histogram (Clipped at 80)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Height Value", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.axvline(np.median(all_valid_heights), color='blue', linestyle='--', label=f"Median: {np.median(all_valid_heights):.2f}")
    ax2.axvline(np.mean(all_valid_heights), color='green', linestyle='-', label=f"Mean: {np.mean(all_valid_heights):.2f}")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    dist_plot_path = os.path.join(out_vis_dir, "height_distribution_audit.png")
    plt.savefig(dist_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved height distribution plot: {dist_plot_path}")

    # Semantic Class Height Sanity Summary
    class_sanity_summary = {}
    for cid, name in CLASS_NAMES.items():
        vals = np.array(class_height_values[cid])
        if len(vals) > 0:
            class_sanity_summary[name] = {
                'class_id': cid,
                'pixel_count': int(global_class_counts[cid]),
                'sampled_pixels': len(vals),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'p95': float(np.percentile(vals, 95)),
                'neg_pct': float((vals < 0).sum() / len(vals) * 100),
                'zero_pct': float((vals == 0).sum() / len(vals) * 100)
            }
        else:
            class_sanity_summary[name] = {
                'class_id': cid,
                'pixel_count': int(global_class_counts[cid]),
                'sampled_pixels': 0,
                'min': None, 'max': None, 'mean': None, 'median': None, 'std': None, 'p95': None,
                'neg_pct': 0, 'zero_pct': 0
            }

    # Save summary dictionary
    summary_report = {
        'total_samples_audited': len(triplets),
        'splits_audited': list(set(t['split'] for t in triplets)),
        'cities_audited': list(set(t['city'] for t in triplets)),
        'pixel_alignment_all_pass': all(r['aligned'] for r in pixel_alignment_results),
        'semantic_class_counts': global_class_counts,
        'semantic_unknown_classes': global_unknown_classes,
        'height_buckets': buckets,
        'height_bucket_percentages': bucket_pcts,
        'class_height_sanity': class_sanity_summary,
        'detailed_h5_samples': detailed_h5_audit
    }
    
    with open(os.path.join(metadata_dir, "audit_summary_metrics.json"), 'w') as f:
        json.dump(summary_report, f, indent=2)
        
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY RESULTS")
    print("=" * 70)
    print(f"Total Samples Audited: {len(triplets)}")
    print(f"Pixel Alignment: {'ALL PASS (100%)' if summary_report['pixel_alignment_all_pass'] else 'FAILURES DETECTED'}")
    
    print("\n--- Semantic Class Distribution ---")
    tot_cls_pixels = sum(global_class_counts.values())
    for cid, count in sorted(global_class_counts.items()):
        pct = (count / tot_cls_pixels) * 100 if tot_cls_pixels > 0 else 0
        print(f"  Class {cid} ({CLASS_NAMES[cid]}): {count:,} pixels ({pct:.2f}%)")
    if global_unknown_classes:
        print(f"  UNKNOWN Classes Found: {global_unknown_classes}")
    else:
        print("  Unknown Classes: NONE (Strictly matches GAMUS 0-6 ontology)")

    print("\n--- Height Value Sanity per Semantic Class ---")
    print(f"{'Class Name':<18} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Median':<8} | {'p95':<8} | {'% < 0':<8}")
    print("-" * 75)
    for name, s in class_sanity_summary.items():
        if s['min'] is not None:
            print(f"{name:<18} | {s['min']:<8.2f} | {s['max']:<8.2f} | {s['mean']:<8.2f} | {s['median']:<8.2f} | {s['p95']:<8.2f} | {s['neg_pct']:<8.2f}%")
        else:
            print(f"{name:<18} | N/A")

    print("\n--- Height Distribution Buckets ---")
    for b_name, pct in bucket_pcts.items():
        print(f"  Bucket {b_name:<15}: {pct:.2f}% ({buckets[b_name]:,} pixels)")
        
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_full_audit()
