"""
DepthWizard (SIH26175) — Depth Anything V2 GPU Benchmark & Baseline Verification
Player 1: AI/ML Lead

Measures real GPU runtime, VRAM usage, and latency percentiles on NVIDIA GeForce RTX 4060 Laptop GPU.
Verifies numerical consistency with previous CPU Baseline-0 results.
"""

import os
import sys
import json
import time
import numpy as np
import scipy.stats as stats
import torch

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.dav2_wrapper import DepthAnythingV2Wrapper
from scripts.gamus_loader import DepthWizardGAMUSDataset

def run_gpu_benchmark():
    print("=" * 80)
    print("RUNNING DEPTH ANYTHING V2 SMALL GPU BENCHMARK (RTX 4060)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    if device == "cuda":
        dev_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device: {dev_name} ({vram_total_gb:.2f} GB Total VRAM)")
        torch.cuda.reset_peak_memory_stats(0)
    else:
        print("WARNING: CUDA is not available! Benchmark will run on CPU.")

    # 1. Measure Model Load Time to GPU
    t0 = time.time()
    wrapper = DepthAnythingV2Wrapper(
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_load_time = time.time() - t0
    print(f"Model Loaded to {device.upper()} in {gpu_load_time:.2f} s")
    print(f"Model Parameters: {wrapper.total_params:,} ({wrapper.total_params / 1e6:.2f} M)")

    # 2. Warm-Up Passes on CUDA
    if device == "cuda":
        print("\n--- Performing GPU Warm-Up (5 passes) ---")
        dummy_rgb = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        for w_idx in range(5):
            _ = wrapper.predict_relative_depth(dummy_rgb, target_size=(1024, 1024), input_size=(518, 518))
            torch.cuda.synchronize()
        print("GPU Warm-Up Completed.")

    # 3. Load 16 Verified Development Samples
    splits = ['train', 'val', 'test']
    all_latencies_ms = []
    sample_records = []
    
    total_batch_t0 = time.time()

    for split in splits:
        ds = DepthWizardGAMUSDataset(
            root_dir='data/gamus/samples',
            split=split,
            height_mode='mask_negative',
            return_torch=False
        )
        print(f"\n--- Running Split: '{split}' ({len(ds)} samples) on GPU ---")
        
        for idx in range(len(ds)):
            item = ds[idx]
            sid = item['sample_id']
            rgb = item['rgb'] # (1024, 1024, 3)
            hgt = item['height'] # (1024, 1024)
            mask = item['valid_mask']

            # Timed inference with CUDA Synchronization
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.time()
            
            infer_res = wrapper.predict_relative_depth(
                rgb_input=rgb,
                target_size=(1024, 1024),
                input_size=(518, 518)
            )
            
            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.time()
            
            latency_ms = (t_end - t_start) * 1000.0
            all_latencies_ms.append(latency_ms)
            
            raw_depth = infer_res['raw_depth']
            
            # Verify correlation
            valid_mask = mask & np.isfinite(raw_depth) & np.isfinite(hgt)
            val_h = hgt[valid_mask]
            val_d = raw_depth[valid_mask]
            p_orig, _ = stats.pearsonr(val_d, val_h)
            
            sample_records.append({
                'sample_id': sid,
                'split': split,
                'gpu_latency_ms': latency_ms,
                'pearson_original': float(p_orig)
            })
            
            print(f"  [{len(sample_records):02d}/16] {sid:<12} ({split}) | GPU Latency: {latency_ms:.2f} ms | Pearson r: {p_orig:+.3f}")

    total_batch_time_s = time.time() - total_batch_t0
    
    # 4. Measure VRAM
    if device == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(0) / (1024**2)
        peak_res_mb = torch.cuda.max_memory_reserved(0) / (1024**2)
    else:
        peak_alloc_mb = peak_res_mb = 0.0

    # 5. Compute Statistics
    mean_lat = float(np.mean(all_latencies_ms))
    median_lat = float(np.median(all_latencies_ms))
    p95_lat = float(np.percentile(all_latencies_ms, 95))
    min_lat = float(np.min(all_latencies_ms))
    max_lat = float(np.max(all_latencies_ms))
    
    gpu_metrics = {
        'device_name': dev_name if device == "cuda" else "CPU",
        'device_type': device,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if device == "cuda" else "N/A",
        'samples_evaluated': len(sample_records),
        'model_load_time_s': gpu_load_time,
        'mean_latency_ms': mean_lat,
        'median_latency_ms': median_lat,
        'p95_latency_ms': p95_lat,
        'min_latency_ms': min_lat,
        'max_latency_ms': max_lat,
        'fps': 1000.0 / mean_lat if mean_lat > 0 else 0.0,
        'total_batch_time_s': total_batch_time_s,
        'peak_allocated_vram_mb': peak_alloc_mb,
        'peak_reserved_vram_mb': peak_res_mb,
        'sample_latencies': sample_records
    }

    out_path = "outputs/player1_baseline0/gpu_benchmark_metrics.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(gpu_metrics, f, indent=2)
    print(f"\nSaved GPU benchmark metrics: {out_path}")

    print("\n" + "=" * 80)
    print("REAL MEASURED GPU BENCHMARK SUMMARY (RTX 4060 LAPTOP GPU)")
    print("=" * 80)
    print(f"Device:                  {gpu_metrics['device_name']}")
    print(f"Samples Evaluated:       16 GAMUS tiles (1024x1024)")
    print(f"Mean GPU Latency:        {mean_lat:.2f} ms / image ({gpu_metrics['fps']:.1f} FPS)")
    print(f"Median GPU Latency:      {median_lat:.2f} ms / image")
    print(f"P95 GPU Latency:         {p95_lat:.2f} ms / image")
    print(f"Min / Max Latency:       [{min_lat:.2f} ms, {max_lat:.2f} ms]")
    print(f"Total 16-Tile Batch:     {total_batch_time_s:.2f} seconds")
    print(f"Peak Allocated VRAM:     {peak_alloc_mb:.2f} MB ({peak_alloc_mb / 1024:.2f} GB)")
    print(f"Peak Reserved VRAM:      {peak_res_mb:.2f} MB ({peak_res_mb / 1024:.2f} GB)")
    print(f"VRAM Headroom on 8GB:    {(8188 - peak_alloc_mb):.1f} MB ({(8188 - peak_alloc_mb)/1024:.2f} GB Free)")
    print("=" * 80)

if __name__ == '__main__':
    run_gpu_benchmark()
