"""
DepthWizard (SIH26175) — Baseline-0: Pretrained Monocular Depth Analysis
Player 1: AI/ML Lead

Evaluates Depth Anything V2 Small on 16 GAMUS remote sensing development samples.
Computes Pearson & Spearman structural correlations, class-wise analysis, height-bucket analysis,
diagnostic affine fitting, and generates 6-panel composite visual comparisons.
"""

import os
import sys
import json
import csv
import time
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.dav2_wrapper import DepthAnythingV2Wrapper
from scripts.gamus_loader import DepthWizardGAMUSDataset

CLASS_NAMES = {
    0: 'Others/Background',
    1: 'Ground',
    2: 'Low vegetation',
    3: 'Buildings',
    4: 'Water',
    5: 'Road',
    6: 'Tree'
}

CLASS_COLORS = {
    0: '#000000', # Black / Others
    1: '#808080', # Gray / Ground
    2: '#90EE90', # Light Green / Low Veg
    3: '#FF0000', # Red / Buildings
    4: '#0000FF', # Blue / Water
    5: '#FFFF00', # Yellow / Road
    6: '#006400', # Dark Green / Tree
}

def robust_normalize(arr, mask=None, p_low=1.0, p_high=99.0):
    """Robust percentile normalization to [0, 1]."""
    if mask is not None and mask.sum() > 0:
        valid_vals = arr[mask]
    else:
        valid_vals = arr[np.isfinite(arr)]
        
    if len(valid_vals) == 0:
        return np.zeros_like(arr)
        
    vmin = np.percentile(valid_vals, p_low)
    vmax = np.percentile(valid_vals, p_high)
    
    if vmax <= vmin:
        return np.zeros_like(arr)
        
    normed = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return normed

def run_baseline0():
    print("=" * 80)
    print("RUNNING PLAYER 1 BASELINE-0: PRETRAINED MONOCULAR DEPTH ANALYSIS (DAV2-SMALL)")
    print("=" * 80)
    
    out_dir = "outputs/player1_baseline0"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Model
    print("Loading Depth Anything V2 Small relative-depth model...")
    model_wrapper = DepthAnythingV2Wrapper(
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu"
    )
    print(f"Model Loaded: {model_wrapper.model_id}")
    print(f"Total Parameters: {model_wrapper.total_params:,} ({model_wrapper.total_params / 1e6:.2f} M)")
    print(f"Load Time: {model_wrapper.load_time:.2f} s")

    # 2. Load Verified GAMUS Development Dataset (16 samples across train, val, test)
    splits = ['train', 'val', 'test']
    all_sample_results = []
    
    # Global class-wise accumulators
    global_class_d_orig = {cid: [] for cid in CLASS_NAMES}
    global_class_d_best = {cid: [] for cid in CLASS_NAMES}
    global_class_hgt = {cid: [] for cid in CLASS_NAMES}
    
    # Global height-bucket accumulators
    bucket_definitions = {
        '0-2m': (0.0, 2.0),
        '2-10m': (2.0, 10.0),
        '10-20m': (10.0, 20.0),
        '20-50m': (20.0, 50.0),
    }
    bucket_d_orig = {b: [] for b in bucket_definitions}
    bucket_d_norm = {b: [] for b in bucket_definitions}
    bucket_hgt = {b: [] for b in bucket_definitions}
    
    total_infer_time_ms = 0.0
    sample_counter = 0

    for split in splits:
        ds = DepthWizardGAMUSDataset(
            root_dir='data/gamus/samples',
            split=split,
            height_mode='mask_negative', # Preserves raw height, returns valid_mask
            return_torch=False
        )
        print(f"\n--- Processing Split: '{split}' ({len(ds)} samples) ---")
        
        for idx in range(len(ds)):
            sample_counter += 1
            item = ds[idx]
            sid = item['sample_id']
            rgb = item['rgb'] # (1024, 1024, 3) float32 in [0, 1]
            hgt = item['height'] # (1024, 1024) float32 metres
            cls = item['semantic'] # (1024, 1024) int64
            mask = item['valid_mask'] # (1024, 1024) bool
            city = sid.split('_')[0]
            
            # 3. Run DAV2 Inference
            infer_res = model_wrapper.predict_relative_depth(
                rgb_input=rgb,
                target_size=(1024, 1024),
                input_size=(518, 518)
            )
            raw_depth = infer_res['raw_depth'] # (1024, 1024) float32
            infer_ms = infer_res['inference_time_ms']
            total_infer_time_ms += infer_ms
            
            # 4. Valid Pixels for Evaluation
            valid_mask = mask & np.isfinite(raw_depth) & np.isfinite(hgt)
            valid_count = int(valid_mask.sum())
            
            val_h = hgt[valid_mask]
            val_d = raw_depth[valid_mask]
            
            # 5. Pearson and Spearman Correlation (Original and Inverted)
            # Pearson Correlation
            if valid_count > 100:
                p_orig, p_pval = stats.pearsonr(val_d, val_h)
                p_inv = -p_orig # Linear inversion: corr(-d, h) = -corr(d, h)
                
                # Spearman Rank Correlation (subsample if large for speed)
                if valid_count > 30000:
                    sub_idx = np.random.choice(valid_count, 30000, replace=False)
                    s_orig, s_pval = stats.spearmanr(val_d[sub_idx], val_h[sub_idx])
                else:
                    s_orig, s_pval = stats.spearmanr(val_d, val_h)
                s_inv = -s_orig
            else:
                p_orig = p_inv = s_orig = s_inv = 0.0

            # 6. Determine Best Orientation
            # If positive correlation, original DAV2 depth increases with height
            # If negative correlation, inverted DAV2 depth increases with height
            if p_orig >= 0:
                best_orientation = "ORIGINAL"
                best_p = p_orig
                best_s = s_orig
                d_best_oriented = raw_depth.copy()
            else:
                best_orientation = "INVERTED"
                best_p = p_inv
                best_s = s_inv
                d_best_oriented = -raw_depth.copy()

            # 7. Normalized Representations for Comparison
            h_norm = robust_normalize(hgt, mask=valid_mask)
            d_orig_norm = robust_normalize(raw_depth, mask=valid_mask)
            d_best_norm = robust_normalize(d_best_oriented, mask=valid_mask)
            diff_map = np.abs(h_norm - d_best_norm)
            diff_map[~valid_mask] = 0.0

            # 8. Diagnostic Affine Calibration (DIAGNOSTIC ONLY)
            if valid_count > 100:
                # Fit h = slope * d + intercept on this single sample
                slope, intercept, r_val, p_val, std_err = stats.linregress(val_d, val_h)
                h_affine = slope * val_d + intercept
                diag_rmse = float(np.sqrt(np.mean((val_h - h_affine)**2)))
                diag_mae = float(np.mean(np.abs(val_h - h_affine)))
            else:
                slope = intercept = diag_rmse = diag_mae = 0.0

            # 9. Class-Wise and Bucket Accumulation
            for cid in CLASS_NAMES:
                c_mask = valid_mask & (cls == cid)
                if c_mask.sum() > 0:
                    sub_n = min(c_mask.sum(), 10000)
                    c_idx = np.random.choice(c_mask.sum(), sub_n, replace=False)
                    global_class_d_orig[cid].extend(raw_depth[c_mask][c_idx].tolist())
                    global_class_d_best[cid].extend(d_best_oriented[c_mask][c_idx].tolist())
                    global_class_hgt[cid].extend(hgt[c_mask][c_idx].tolist())

            for b_name, (b_min, b_max) in bucket_definitions.items():
                b_mask = valid_mask & (hgt >= b_min) & (hgt < b_max)
                if b_mask.sum() > 0:
                    sub_n = min(b_mask.sum(), 10000)
                    b_idx = np.random.choice(b_mask.sum(), sub_n, replace=False)
                    bucket_d_orig[b_name].extend(raw_depth[b_mask][b_idx].tolist())
                    bucket_d_norm[b_name].extend(d_best_norm[b_mask][b_idx].tolist())
                    bucket_hgt[b_name].extend(hgt[b_mask][b_idx].tolist())

            # Record sample result
            sample_record = {
                'sample_id': sid,
                'split': split,
                'city': city,
                'inference_time_ms': float(infer_ms),
                'valid_pixel_count': valid_count,
                'agl_min_m': float(np.min(val_h)) if len(val_h)>0 else 0.0,
                'agl_max_m': float(np.max(val_h)) if len(val_h)>0 else 0.0,
                'agl_mean_m': float(np.mean(val_h)) if len(val_h)>0 else 0.0,
                'agl_median_m': float(np.median(val_h)) if len(val_h)>0 else 0.0,
                'dav2_raw_min': float(np.min(raw_depth)),
                'dav2_raw_max': float(np.max(raw_depth)),
                'dav2_raw_mean': float(np.mean(raw_depth)),
                'dav2_raw_std': float(np.std(raw_depth)),
                'best_orientation': best_orientation,
                'pearson_original': float(p_orig),
                'pearson_inverted': float(p_inv),
                'pearson_best': float(best_p),
                'spearman_original': float(s_orig),
                'spearman_inverted': float(s_inv),
                'spearman_best': float(best_s),
                'diagnostic_affine_slope': float(slope),
                'diagnostic_affine_intercept': float(intercept),
                'diagnostic_affine_rmse_m': float(diag_rmse),
                'diagnostic_affine_mae_m': float(diag_mae)
            }
            all_sample_results.append(sample_record)

            print(f"[{sample_counter}/16] {sid:<12} ({split}, {city}): Time={infer_ms:.1f}ms | Orient={best_orientation:<8} | Pearson(orig)={p_orig:+.3f} | Pearson(inv)={p_inv:+.3f} | Spearman(best)={best_s:+.3f}")

            # 10. Generate 6-Panel Composite Inspection Figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
            
            # Panel 1: RGB Optical
            axes[0, 0].imshow(np.clip(rgb, 0.0, 1.0))
            axes[0, 0].set_title(f"1. Optical RGB Satellite Image\nSize: (1024, 1024, 3)", fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')

            # Panel 2: GAMUS Metric AGL
            h_disp = np.copy(hgt)
            h_disp[~valid_mask] = np.nan
            im_h = axes[0, 1].imshow(h_disp, cmap='turbo', vmin=0, vmax=max(25.0, np.percentile(val_h, 98) if len(val_h)>0 else 25.0))
            cbar_h = fig.colorbar(im_h, ax=axes[0, 1], fraction=0.046, pad=0.04)
            cbar_h.set_label('Metric AGL Height (m)', fontsize=9)
            axes[0, 1].set_title(f"2. GAMUS Ground Truth (AGL)\nRange: [{sample_record['agl_min_m']:.1f}, {sample_record['agl_max_m']:.1f}]m | Mean: {sample_record['agl_mean_m']:.1f}m", fontsize=10, fontweight='bold')
            axes[0, 1].axis('off')

            # Panel 3: DAV2 Raw Depth
            im_d = axes[0, 2].imshow(raw_depth, cmap='viridis')
            cbar_d = fig.colorbar(im_d, ax=axes[0, 2], fraction=0.046, pad=0.04)
            cbar_d.set_label('Scale-Agnostic Output', fontsize=9)
            axes[0, 2].set_title(f"3. DAV2 Raw Relative Depth\nRange: [{sample_record['dav2_raw_min']:.2f}, {sample_record['dav2_raw_max']:.2f}] | Mean: {sample_record['dav2_raw_mean']:.2f}", fontsize=10, fontweight='bold')
            axes[0, 2].axis('off')

            # Panel 4: Best-Orientation Normalized Depth
            im_bn = axes[1, 0].imshow(d_best_norm, cmap='turbo', vmin=0.0, vmax=1.0)
            cbar_bn = fig.colorbar(im_bn, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar_bn.set_label('Normalized [0, 1]', fontsize=9)
            axes[1, 0].set_title(f"4. Normalized DAV2 ({best_orientation})\nPearson r: {best_p:+.3f} | Spearman ρ: {best_s:+.3f}", fontsize=10, fontweight='bold')
            axes[1, 0].axis('off')

            # Panel 5: Absolute Normalized Difference
            im_diff = axes[1, 1].imshow(diff_map, cmap='magma', vmin=0.0, vmax=1.0)
            cbar_diff = fig.colorbar(im_diff, ax=axes[1, 1], fraction=0.046, pad=0.04)
            cbar_diff.set_label('Absolute Error (|H - D|)', fontsize=9)
            axes[1, 1].set_title(f"5. Absolute Structural Difference\nMean Norm Diff: {np.mean(diff_map[valid_mask]):.3f}", fontsize=10, fontweight='bold')
            axes[1, 1].axis('off')

            # Panel 6: Semantic Mask
            cmap_colors = [CLASS_COLORS[i] for i in range(7)]
            cmap_sem = ListedColormap(cmap_colors)
            norm_sem = BoundaryNorm(range(8), cmap_sem.N)
            axes[1, 2].imshow(cls, cmap=cmap_sem, norm=norm_sem, interpolation='nearest')
            axes[1, 2].set_title("6. GAMUS Semantic Classes", fontsize=10, fontweight='bold')
            axes[1, 2].axis('off')
            
            u_classes = np.unique(cls[valid_mask])
            patches = [mpatches.Patch(color=CLASS_COLORS[int(c)], label=f"{int(c)}: {CLASS_NAMES[int(c)]}") for c in u_classes if int(c) in CLASS_COLORS]
            axes[1, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True)

            fig.suptitle(f"DepthWizard Baseline-0 | Sample: {sid} | Split: {split.upper()} | City: {city} | Infer Time: {infer_ms:.1f}ms\n"
                         f"Pearson r (orig={p_orig:+.3f}, inv={p_inv:+.3f}) | Spearman ρ (best={best_s:+.3f}) | Diag Affine RMSE: {diag_rmse:.2f}m",
                         fontsize=12, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            fig_path = os.path.join(out_dir, f"{split}_{sid}_dav2_baseline.png")
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)

    # 11. Class-Wise Analysis Summary
    class_wise_summary = {}
    print("\n" + "=" * 80)
    print("CLASS-WISE STRUCTURAL ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Class Name':<18} | {'Pixels':<8} | {'Mean AGL (m)':<12} | {'Mean DAV2':<10} | {'Pearson (orig)':<15} | {'Spearman (best)'}")
    print("-" * 80)
    
    for cid, cname in CLASS_NAMES.items():
        c_hgt = np.array(global_class_hgt[cid])
        c_d_orig = np.array(global_class_d_orig[cid])
        c_d_best = np.array(global_class_d_best[cid])
        
        if len(c_hgt) > 100:
            r_orig, _ = stats.pearsonr(c_d_orig, c_hgt)
            r_inv = -r_orig
            # Subsample for spearman
            sub_k = min(len(c_hgt), 20000)
            sub_sel = np.random.choice(len(c_hgt), sub_k, replace=False)
            s_best, _ = stats.spearmanr(c_d_best[sub_sel], c_hgt[sub_sel])
            
            mean_h = float(np.mean(c_hgt))
            mean_d = float(np.mean(c_d_orig))
            
            class_wise_summary[cname] = {
                'class_id': cid,
                'pixel_count': len(c_hgt),
                'mean_agl_m': mean_h,
                'mean_dav2_raw': mean_d,
                'pearson_original': float(r_orig),
                'pearson_inverted': float(r_inv),
                'spearman_best': float(s_best)
            }
            print(f"{cname:<18} | {len(c_hgt):<8} | {mean_h:<12.2f} | {mean_d:<10.2f} | {r_orig:<+15.3f} | {s_best:<+.3f}")
        else:
            class_wise_summary[cname] = {
                'class_id': cid, 'pixel_count': len(c_hgt), 'mean_agl_m': 0.0, 'mean_dav2_raw': 0.0,
                'pearson_original': 0.0, 'pearson_inverted': 0.0, 'spearman_best': 0.0
            }
            print(f"{cname:<18} | {len(c_hgt):<8} | N/A")

    # 12. Height-Bucket Analysis Summary
    bucket_summary = {}
    print("\n" + "=" * 80)
    print("HEIGHT-BUCKET STRUCTURAL RESPONSE ANALYSIS")
    print("=" * 80)
    print(f"{'Height Bucket':<15} | {'Pixel Count':<12} | {'Mean AGL (m)':<12} | {'Mean DAV2 Raw':<14} | {'Mean DAV2 Norm'}")
    print("-" * 80)
    for b_name in bucket_definitions:
        b_h = np.array(bucket_hgt[b_name])
        b_d = np.array(bucket_d_orig[b_name])
        b_dn = np.array(bucket_d_norm[b_name])
        
        if len(b_h) > 0:
            m_h = float(np.mean(b_h))
            m_d = float(np.mean(b_d))
            m_dn = float(np.mean(b_dn))
            bucket_summary[b_name] = {
                'pixel_count': len(b_h),
                'mean_agl_m': m_h,
                'mean_dav2_raw': m_d,
                'mean_dav2_norm': m_dn
            }
            print(f"{b_name:<15} | {len(b_h):<12} | {m_h:<12.2f} | {m_d:<14.2f} | {m_dn:<.3f}")
        else:
            bucket_summary[b_name] = {'pixel_count': 0, 'mean_agl_m': 0.0, 'mean_dav2_raw': 0.0, 'mean_dav2_norm': 0.0}

    # 13. Aggregate Overall Metrics
    all_p_orig = [r['pearson_original'] for r in all_sample_results]
    all_p_inv = [r['pearson_inverted'] for r in all_sample_results]
    all_p_best = [r['pearson_best'] for r in all_sample_results]
    all_s_best = [r['spearman_best'] for r in all_sample_results]
    all_times = [r['inference_time_ms'] for r in all_sample_results]
    all_diag_rmse = [r['diagnostic_affine_rmse_m'] for r in all_sample_results]
    all_diag_mae = [r['diagnostic_affine_mae_m'] for r in all_sample_results]
    
    overall_summary = {
        'model_name': 'Depth Anything V2 Small (Relative Depth)',
        'model_id': model_wrapper.model_id,
        'model_params': model_wrapper.total_params,
        'samples_evaluated': len(all_sample_results),
        'mean_inference_time_ms': float(np.mean(all_times)),
        'total_inference_time_s': float(sum(all_times)/1000.0),
        'mean_pearson_original': float(np.mean(all_p_orig)),
        'mean_pearson_inverted': float(np.mean(all_p_inv)),
        'mean_pearson_best': float(np.mean(all_p_best)),
        'median_pearson_best': float(np.median(all_p_best)),
        'mean_spearman_best': float(np.mean(all_s_best)),
        'median_spearman_best': float(np.median(all_s_best)),
        'mean_diagnostic_affine_rmse_m': float(np.mean(all_diag_rmse)),
        'mean_diagnostic_affine_mae_m': float(np.mean(all_diag_mae)),
        'sample_metrics': all_sample_results,
        'class_wise_metrics': class_wise_summary,
        'height_bucket_metrics': bucket_summary
    }

    # Save Machine-Readable JSON
    json_path = os.path.join(out_dir, "baseline0_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\nSaved machine-readable JSON metrics: {json_path}")

    # Save Machine-Readable CSV
    csv_path = os.path.join(out_dir, "baseline0_metrics.csv")
    if all_sample_results:
        keys = list(all_sample_results[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_sample_results:
                writer.writerow(r)
    print(f"Saved machine-readable CSV metrics: {csv_path}")

    print("\n" + "=" * 80)
    print("BASELINE-0 OVERALL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Samples Evaluated: {len(all_sample_results)}")
    print(f"Mean Inference Time: {overall_summary['mean_inference_time_ms']:.1f} ms / image")
    print(f"Mean Pearson Correlation (Original DAV2): {overall_summary['mean_pearson_original']:+.3f}")
    print(f"Mean Pearson Correlation (Inverted DAV2): {overall_summary['mean_pearson_inverted']:+.3f}")
    print(f"Mean Pearson Correlation (Best Oriented): {overall_summary['mean_pearson_best']:+.3f}")
    print(f"Mean Spearman Rank Correlation:          {overall_summary['mean_spearman_best']:+.3f}")
    print(f"Diagnostic Affine Fit RMSE (Single Sample Fit): {overall_summary['mean_diagnostic_affine_rmse_m']:.2f} m")
    print(f"Diagnostic Affine Fit MAE (Single Sample Fit):  {overall_summary['mean_diagnostic_affine_mae_m']:.2f} m")
    print("=" * 80)

if __name__ == '__main__':
    run_baseline0()
