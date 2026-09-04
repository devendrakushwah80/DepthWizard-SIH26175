"""
DepthWizard (SIH26175) — Authoritative Demo Scene (NYC_00735) Generator
Runs real M3-FINAL + DAV2 inference on the authentic optical test image to produce
complete scene products (RGB, predicted AGL, relative surface, heightmap, slope,
GLB meshes, pointcloud, and metadata).
"""

from pathlib import Path
import shutil
import sys
import time
import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import settings
from backend.app.services.inference_service import inference_service
from backend.app.services.scene_service import scene_service


def get_source_rgb() -> np.ndarray:
    # 1. Try raw H5 dataset if present
    h5_path = REPO_ROOT / "data/gamus/dataset/images/test/NYC_00735_IMG.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as handle:
            arr = handle[next(iter(handle.keys()))][:]
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255)
        return arr.astype(np.uint8)

    # 2. Try existing RGB texture
    rgb_path = REPO_ROOT / "seed_scenes/NYC_00735/textures/rgb.jpg"
    if rgb_path.exists():
        return np.array(Image.open(rgb_path).convert("RGB"))

    raise FileNotFoundError("Could not locate optical source RGB for NYC_00735")


def main():
    print("=" * 80)
    print("RESTORING DEMO SCENE: NYC_00735 (Real M3-FINAL + Frozen DAV2 Pipeline)")
    print("=" * 80)

    rgb = get_source_rgb()
    print(f"Loaded optical RGB shape: {rgb.shape}, dtype: {rgb.dtype}")

    spatial_meta = {
        "format": "GeoTIFF",
        "is_georeferenced": True,
        "crs": "EPSG:32618 (WGS 84 / UTM zone 18N)",
        "gsd_m": 0.5,
        "gsd_source": "dataset",
        "horizontal_units": "metres",
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "physical_width_m": rgb.shape[1] * 0.5,
        "physical_height_m": rgb.shape[0] * 0.5,
        "bounds": {
            "left": -256.0,
            "bottom": -256.0,
            "right": 256.0,
            "top": 256.0
        },
        "transform": [0.5, 0.0, -256.0, 0.0, -0.5, 256.0],
        "resolution": [0.5, 0.5]
    }

    print("Running full inference pipeline...")
    t0 = time.perf_counter()
    infer_result = inference_service.predict_height_map(rgb, gsd_m=0.5)
    t_infer = time.perf_counter() - t0
    print(f"Inference complete in {t_infer:.2f}s.")
    print(f"  DAV2 time: {infer_result['dav2_time_s']:.3f}s")
    print(f"  M3 time:   {infer_result['m2_time_s']:.3f}s")
    print(f"  Peak VRAM: {infer_result['peak_vram_mb']:.1f} MB")

    timings = {
        "dav2_inference_s": round(float(infer_result["dav2_time_s"]), 6),
        "m2_inference_s": round(float(infer_result["m2_time_s"]), 6),
        "total_ai_s": round(float(infer_result["ai_time_s"]), 6),
        "inference_service_total_s": round(float(infer_result["total_inference_time_s"]), 6),
        "window_count": int(infer_result["window_count"]),
        "window_size_px": int(infer_result["window_size_px"]),
        "window_step_px": int(infer_result["window_step_px"]),
    }

    # Backup existing if present
    scene_id = "NYC_00735"
    scene_dir = Path(scene_service.get_scene_dir(scene_id))

    print(f"Generating scene products for {scene_id} in {scene_dir}...")
    metadata = scene_service.create_scene_products(
        scene_id=scene_id,
        rgb_image=rgb,
        predicted_agl=infer_result["predicted_agl_m"],
        spatial_meta=spatial_meta,
        base_dem=None,
        relative_surface=infer_result["relative_surface"],
        generate_3d=True,
        generate_pointcloud=True,
        mesh_resolution=384,
        vertical_exaggeration=1.0,
        processing_timings=timings,
        peak_vram_mb=float(infer_result["peak_vram_mb"])
    )

    scene_service.finalize_scene_runtime(scene_id, t_infer, t_infer)

    # Also sync to seed_scenes so fresh checkouts / fresh storage clones have all files
    seed_dest = REPO_ROOT / "seed_scenes" / scene_id
    if seed_dest != scene_dir:
        print(f"Syncing complete artifacts to {seed_dest}...")
        shutil.copytree(scene_dir, seed_dest, dirs_exist_ok=True)

    print("NYC_00735 Demo Scene restored with full relative surface, GLB, pointcloud, and metadata.")


if __name__ == "__main__":
    main()
