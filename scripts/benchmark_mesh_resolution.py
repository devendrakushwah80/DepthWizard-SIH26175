"""
DepthWizard (SIH26175) — Mesh Resolution & Geometry Fidelity Benchmark
Player 3: 3D Reconstruction & Visualization Lead

Systematically benchmarks multiple mesh grid resolutions (256, 384, 512, 1024)
and downsampling algorithms to find the optimal balance between geometry fidelity,
height preservation, file size, and browser rendering efficiency.
"""

import os
import sys
import time
import json
import csv
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.geometry.scene import Scene3D
from src.geometry.raster_to_mesh import build_terrain_mesh, downsample_raster
from src.geometry.gltf_export import export_mesh_glb

def benchmark_scene(scene: Scene3D, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    resolutions = [256, 384, 512]
    methods = ["bilinear", "area", "max_aware"]
    
    full_h = scene.predicted_agl # 1024x1024 reference
    max_true_h = float(np.max(full_h))
    
    benchmark_records = []
    
    for res in resolutions:
        for method in methods:
            t0 = time.time()
            
            # 1. Downsample raster
            down_h = downsample_raster(full_h, (res, res), method=method)
            down_time = time.time() - t0
            
            # 2. Measure Height Reconstruction Error (upsample back to 1024x1024 to compare pixel-for-pixel)
            upsampled = scipy.ndimage.zoom(down_h, (1024 / res, 1024 / res), order=1, mode='nearest')
            diff = np.abs(upsampled - full_h)
            
            mae_m = float(np.mean(diff))
            rmse_m = float(np.sqrt(np.mean(diff ** 2)))
            max_err_m = float(np.max(diff))
            p95_err_m = float(np.percentile(diff, 95))
            
            peak_h = float(np.max(down_h))
            peak_retention_pct = (peak_h / max(1e-4, max_true_h)) * 100.0
            
            # 3. Build Mesh
            t_mesh0 = time.time()
            mesh = build_terrain_mesh(scene, mesh_resolution=res, downsample_method=method, vertical_exaggeration=1.0, add_side_skirts=True)
            mesh_time = time.time() - t_mesh0
            
            # 4. Export GLB
            glb_path = os.path.join(out_dir, f"{scene.scene_id}_{res}_{method}.glb")
            glb_meta = export_mesh_glb(mesh, glb_path)
            
            # 5. Theoretical Browser Performance Estimate (based on WebGL triangle budget)
            # Modern GPUs easily achieve 60 FPS under 500k triangles
            tri_count = mesh.triangle_count
            if tri_count < 150000:
                est_fps = "60+ FPS (Ultra Smooth)"
                perf_tier = "A+"
            elif tri_count < 350000:
                est_fps = "60 FPS (Smooth / Optimal)"
                perf_tier = "A"
            elif tri_count < 600000:
                est_fps = "45-60 FPS (High Fidelity)"
                perf_tier = "B"
            else:
                est_fps = "30-45 FPS (Heavy)"
                perf_tier = "C"

            record = {
                'scene_id': scene.scene_id,
                'resolution': f"{res}x{res}",
                'resolution_n': res,
                'downsample_method': method,
                'vertex_count': mesh.vertex_count,
                'triangle_count': mesh.triangle_count,
                'glb_size_mb': float(glb_meta['file_size_mb']),
                'height_mae_m': mae_m,
                'height_rmse_m': rmse_m,
                'height_max_err_m': max_err_m,
                'p95_error_m': p95_err_m,
                'peak_retention_pct': peak_retention_pct,
                'generation_time_ms': float((down_time + mesh_time) * 1000.0),
                'estimated_browser_fps': est_fps,
                'performance_tier': perf_tier
            }
            benchmark_records.append(record)
            print(f"  Res: {res}x{res} | Method: {method:9s} | Verts: {mesh.vertex_count:,} | Tris: {mesh.triangle_count:,} | Size: {glb_meta['file_size_mb']:.2f}MB | MAE: {mae_m:.3f}m | Peak: {peak_retention_pct:.1f}%")

    # Save CSV
    csv_path = os.path.join(out_dir, "mesh_resolution_benchmark.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_records[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_records)

    # Save JSON
    json_path = os.path.join(out_dir, "mesh_resolution_benchmark.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'benchmark_results': benchmark_records}, f, indent=2)

    # Generate Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # 1. Triangles vs File Size
    res_labels = [f"{r['resolution']}\n({r['downsample_method']})" for r in benchmark_records]
    tris = [r['triangle_count'] / 1000.0 for r in benchmark_records]
    sizes = [r['glb_size_mb'] for r in benchmark_records]
    maes = [r['height_mae_m'] for r in benchmark_records]
    peaks = [r['peak_retention_pct'] for r in benchmark_records]
    
    x = np.arange(len(benchmark_records))
    
    axes[0].bar(x, tris, color='steelblue', alpha=0.85)
    axes[0].set_ylabel('Triangles (Thousands)', fontweight='bold')
    axes[0].set_title('Geometric Complexity (Triangle Count)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(res_labels, rotation=45, ha='right', fontsize=8)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    axes[1].bar(x, sizes, color='coral', alpha=0.85)
    axes[1].set_ylabel('GLB File Size (MB)', fontweight='bold')
    axes[1].set_title('Asset Size for Web Delivery', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(res_labels, rotation=45, ha='right', fontsize=8)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    axes[2].plot(x, maes, marker='o', color='crimson', linewidth=2, label='Reconstruction MAE (m)')
    axes[2].set_ylabel('Reconstruction Error (m)', fontweight='bold', color='crimson')
    axes[2].tick_params(axis='y', labelcolor='crimson')
    axes[2].set_title('Height Fidelity & Downsampling Error', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(res_labels, rotation=45, ha='right', fontsize=8)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    ax2_twin = axes[2].twinx()
    ax2_twin.plot(x, peaks, marker='s', color='forestgreen', linewidth=2, linestyle='--', label='Peak Retention (%)')
    ax2_twin.set_ylabel('Peak Elevation Preserved (%)', fontweight='bold', color='forestgreen')
    ax2_twin.tick_params(axis='y', labelcolor='forestgreen')
    ax2_twin.set_ylim([70, 105])

    plt.suptitle(f"DepthWizard Phase 3 — Mesh Resolution & Geometric Fidelity Benchmark ({scene.scene_id})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "resolution_comparison_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"\nBenchmark completed. Summary saved to: {csv_path} and {plot_path}")
    return benchmark_records

def main():
    rgb_path = 'data/gamus/dataset/images/test/NYC_00735_IMG.h5'
    hgt_path = 'data/gamus/dataset/heights/test/NYC_00735_AGL.h5'
    cls_path = 'data/gamus/dataset/classes/test/NYC_00735_CLS.h5'

    with h5py.File(rgb_path, 'r') as f:
        k = list(f.keys())[0]
        rgb = f[k][:]
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.max() > 1.5:
            rgb = rgb.astype(np.uint8)

    with h5py.File(hgt_path, 'r') as f:
        k = list(f.keys())[0]
        hgt = f[k][:].astype(np.float32)
        if hgt.ndim == 3:
            hgt = hgt[0] if hgt.shape[0] == 1 else hgt[:, :, 0]

    with h5py.File(cls_path, 'r') as f:
        k = list(f.keys())[0]
        sem = f[k][:].astype(np.int64)
        if sem.ndim == 3:
            sem = sem[0] if sem.shape[0] == 1 else sem[:, :, 0]

    scene = Scene3D(scene_id='NYC_00735', rgb=rgb, predicted_agl=hgt, semantic=sem, gsd=0.5)
    out_dir = 'outputs/player3_3d/benchmarks'
    benchmark_scene(scene, out_dir)

if __name__ == '__main__':
    main()
