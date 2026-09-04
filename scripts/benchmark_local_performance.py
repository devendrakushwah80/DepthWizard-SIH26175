"""
DepthWizard SIH26175 — Local RTX 4060 Performance Benchmark
Measures:
- DAV2 time
- M3 time
- Mesh generation time
- Full pipeline time
- Peak VRAM
For both 512x512 and 1024x1024 inputs.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.inference_service import inference_service
from backend.app.services.scene_service import scene_service
from backend.app.services.model_service import model_service
from src.geometry.scene import Scene3D
from src.geometry.raster_to_mesh import build_terrain_mesh
from src.geometry.gltf_export import export_mesh_glb


def benchmark_resolution(width: int, height: int, name: str):
    print(f"\n--- Benchmarking {name} ({width}x{height}) ---")
    
    # Create realistic RGB input
    rng = np.random.default_rng(42)
    rgb = (rng.uniform(50, 200, (height, width, 3))).astype(np.uint8)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
    t_start = time.perf_counter()
    
    # 1. AI Inference (DAV2 + M3)
    infer_res = inference_service.predict_height_map(rgb, gsd_m=0.5)
    
    dav2_time = infer_res["dav2_time_s"]
    m3_time = infer_res["m2_time_s"]
    pred_agl = infer_res["predicted_agl_m"]
    
    # 2. Mesh Generation (384x384 standard web mesh)
    scene_obj = Scene3D(
        scene_id=f"bench_{width}",
        rgb=rgb,
        predicted_agl=pred_agl,
        gsd=0.5,
        is_georeferenced=True,
    )
    
    t_mesh_start = time.perf_counter()
    mesh_3d = build_terrain_mesh(
        scene_obj,
        mesh_resolution=384,
        downsample_method="max_aware",
        vertical_exaggeration=1.0,
        add_side_skirts=True,
    )
    temp_glb = REPO_ROOT / "outputs" / f"bench_{width}.glb"
    export_mesh_glb(mesh_3d, str(temp_glb))
    if temp_glb.exists():
        temp_glb.unlink()
    mesh_time = time.perf_counter() - t_mesh_start
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0
        
    full_pipeline_time = time.perf_counter() - t_start
    
    print(f"DAV2 time:            {dav2_time:.4f} s")
    print(f"M3 time:              {m3_time:.4f} s")
    print(f"Mesh generation time: {mesh_time:.4f} s")
    print(f"Full pipeline time:   {full_pipeline_time:.4f} s")
    print(f"Peak VRAM:            {peak_vram_mb:.1f} MB")
    
    return {
        "dav2_s": dav2_time,
        "m3_s": m3_time,
        "mesh_s": mesh_time,
        "total_s": full_pipeline_time,
        "vram_mb": peak_vram_mb
    }


def main():
    print("=" * 80)
    print("DEPTHWIZARD LOCAL PERFORMANCE SMOKE TEST (RTX 4060 Laptop GPU)")
    print("=" * 80)
    model_service.ensure_initialized()
    print(f"Device: {model_service.device_name}")
    print(f"Model:  {model_service.model_family}")
    
    # Warmup run
    inference_service.predict_height_map(np.zeros((256, 256, 3), dtype=np.uint8), gsd_m=0.5)
    
    res_512 = benchmark_resolution(512, 512, "512x512")
    res_1024 = benchmark_resolution(1024, 1024, "1024x1024")
    
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"512x512:   DAV2={res_512['dav2_s']:.3f}s | M3={res_512['m3_s']:.3f}s | Mesh={res_512['mesh_s']:.3f}s | Pipeline={res_512['total_s']:.3f}s | VRAM={res_512['vram_mb']:.1f}MB")
    print(f"1024x1024: DAV2={res_1024['dav2_s']:.3f}s | M3={res_1024['m3_s']:.3f}s | Mesh={res_1024['mesh_s']:.3f}s | Pipeline={res_1024['total_s']:.3f}s | VRAM={res_1024['vram_mb']:.1f}MB")


if __name__ == "__main__":
    main()
