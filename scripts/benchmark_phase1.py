"""
DepthWizard (SIH26175) — Performance Benchmark Suite
Benchmarks DAV2 inference, M2 inference, geospatial processing, mesh generation,
GLB export, total latency, peak VRAM, and GLB file size across 3 resolutions:
1. 512x512
2. 1024x1024
3. Non-square: 768x512
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from backend.app.services.model_service import model_service
from backend.app.services.inference_service import InferenceService
from backend.app.services.geospatial_service import GeospatialService
from src.geometry.scene import Scene3D
from src.geometry.raster_to_mesh import build_terrain_mesh
from src.geometry.gltf_export import export_mesh_glb


def get_hardware_info() -> dict:
    cpu_info = platform.processor() or platform.machine()
    gpu_info = "None (CPU mode)"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }


def create_synthetic_geotiff(path: Path, width: int, height: int, res: float = 0.5) -> Path:
    transform = from_origin(500000.0, 4500000.0, res, res)
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack([
        np.clip(xx * (255.0 / width), 0, 255).astype(np.uint8),
        np.clip(yy * (255.0 / height), 0, 255).astype(np.uint8),
        np.full_like(xx, 140, dtype=np.uint8),
    ], axis=0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32618",
        transform=transform,
    ) as dst:
        dst.write(rgb)
    return path


def benchmark_resolution(width: int, height: int, tmp_dir: Path, warmup: bool = False) -> dict:
    tif_path = tmp_dir / f"bench_{width}x{height}.tif"
    create_synthetic_geotiff(tif_path, width, height)

    geo_svc = GeospatialService()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    # 1. Geospatial & IO parsing
    t_geo_0 = time.perf_counter()
    meta = geo_svc.parse_geotiff(str(tif_path))
    rgb_image = meta["rgb_uint8"]
    t_geospatial = time.perf_counter() - t_geo_0

    # 2 & 3. Model Inference (DAV2 prior + M2-FINAL)
    inf_svc = InferenceService()
    inf_res = inf_svc.predict_height_map(rgb_image)
    predicted_agl = inf_res["predicted_agl_m"]
    t_dav2 = inf_res["dav2_time_s"]
    t_m2 = inf_res["m2_time_s"]

    # 4. 3D Scene object construction
    scene_obj = Scene3D(
        scene_id=f"bench_{width}x{height}",
        rgb=rgb_image,
        predicted_agl=predicted_agl,
        gsd=meta.get("gsd_m", 0.5),
        is_georeferenced=True,
        crs=meta.get("crs"),
        transform=meta.get("transform"),
        bounds=meta.get("bounds"),
        base_dem=None,
    )

    # 5. Mesh generation
    t_mesh_0 = time.perf_counter()
    mesh = build_terrain_mesh(
        scene_obj,
        mesh_resolution=min(width, height, 384),
        downsample_method="max_aware",
        vertical_exaggeration=1.0,
        add_side_skirts=True,
    )
    t_mesh = time.perf_counter() - t_mesh_0

    # 6. GLB Export
    glb_out_path = tmp_dir / f"bench_{width}x{height}.glb"
    t_glb_0 = time.perf_counter()
    export_mesh_glb(mesh, str(glb_out_path))
    t_glb = time.perf_counter() - t_glb_0

    t_total = time.perf_counter() - t_start

    # Peak VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    glb_size_bytes = glb_out_path.stat().st_size if glb_out_path.exists() else 0

    return {
        "dimensions": f"{width}x{height}",
        "width_px": width,
        "height_px": height,
        "total_pixels": width * height,
        "dav2_inference_s": round(t_dav2, 4),
        "m2_inference_s": round(t_m2, 4),
        "geospatial_processing_s": round(t_geospatial, 4),
        "mesh_generation_s": round(t_mesh, 4),
        "glb_export_s": round(t_glb, 4),
        "total_processing_latency_s": round(t_total, 4),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "glb_size_bytes": glb_size_bytes,
        "glb_size_mb": round(glb_size_bytes / (1024 * 1024), 2),
    }


def main():
    print("=" * 80)
    print("  ISRO DEPTHWIZARD (SIH26175) — PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)

    hw = get_hardware_info()
    print(f"Platform:       {hw['platform']}")
    print(f"CPU:            {hw['cpu']}")
    print(f"GPU:            {hw['gpu']}")
    print(f"CUDA Available: {hw['cuda_available']}")
    print(f"PyTorch:        {hw['torch_version']}")
    print("=" * 80)

    # Initialize model service (preloads models)
    model_service.initialize()

    tmp_dir = Path("outputs/benchmarks")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Warmup run
    print("\nWarming up GPU inference pipeline...")
    benchmark_resolution(256, 256, tmp_dir, warmup=True)
    print("Warmup complete.\n")

    test_cases = [
        (512, 512, "Square 512x512"),
        (1024, 1024, "Square 1024x1024"),
        (768, 512, "Non-Square 768x512"),
    ]

    results = []
    for w, h, label in test_cases:
        print(f"Running benchmark: {label} ...")
        res = benchmark_resolution(w, h, tmp_dir)
        results.append(res)
        print(f"  -> DAV2 Inference:       {res['dav2_inference_s']:.4f} s")
        print(f"  -> M2 Inference:         {res['m2_inference_s']:.4f} s")
        print(f"  -> Geospatial Parsing:   {res['geospatial_processing_s']:.4f} s")
        print(f"  -> Mesh Generation:      {res['mesh_generation_s']:.4f} s")
        print(f"  -> GLB Export:           {res['glb_export_s']:.4f} s")
        print(f"  -> Total Latency:        {res['total_processing_latency_s']:.4f} s")
        print(f"  -> Peak VRAM:            {res['peak_vram_mb']:.2f} MB")
        print(f"  -> GLB Size:             {res['glb_size_mb']:.2f} MB ({res['glb_size_bytes']:,} bytes)")
        print()

    benchmark_summary = {
        "hardware": hw,
        "benchmarks": results,
    }

    out_file = tmp_dir / "phase1_performance_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("=" * 80)
    print(f"Benchmark results saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
