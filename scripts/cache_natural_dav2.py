"""
Precompute DAV2 priors for the 300 natural training scenes and 136 urban US3D training scenes.
Saves to data/gamus/cache_dav2/{sid}_dav2.npy.
"""
from pathlib import Path
import json
import numpy as np
import rasterio
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "gamus" / "cache_dav2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DAV2_ID = "depth-anything/Depth-Anything-V2-Small-hf"

print(f"Loading DAV2 from {DAV2_ID} on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(DAV2_ID)
model = AutoModelForDepthEstimation.from_pretrained(DAV2_ID).to(DEVICE).eval()

# Load all records
records = {}
with open(PROJECT_ROOT / "data" / "m3" / "manifests" / "all_records.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            records[r["record_id"]] = r

# Load m3d_train IDs
with open(PROJECT_ROOT / "data" / "m3" / "manifests" / "m3d_train.txt", "r", encoding="utf-8") as f:
    train_ids = [line.strip() for line in f if line.strip()]

# Target US3D records in train split
target_records = [
    records[rid] for rid in train_ids 
    if "US3D" in records.get(rid, {}).get("dataset", "")
]

print(f"Total US3D records in m3d_train: {len(target_records)}")

cached_count = 0
generated_count = 0

for idx, rec in enumerate(target_records):
    sid = rec["record_id"].replace("GAMUS_", "").replace("US3D_", "").replace("NATURAL_", "")
    out_path = CACHE_DIR / f"{sid}_dav2.npy"
    if out_path.exists():
        cached_count += 1
        continue

    # Load optical RGB
    with rasterio.open(rec["rgb_path"]) as src:
        rgb = src.read()
        if rgb.shape[0] >= 3:
            rgb_hw = np.transpose(rgb[:3], (1, 2, 0))
        else:
            rgb_hw = np.stack([rgb[0]] * 3, axis=-1)

    pil_img = Image.fromarray(rgb_hw)
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
        depth_raw = outputs.predicted_depth.unsqueeze(1)
        depth_upscaled = torch.nn.functional.interpolate(
            depth_raw, size=rgb_hw.shape[:2], mode="bilinear", align_corners=False
        )
        d_min = depth_upscaled.amin(dim=[2, 3], keepdim=True)
        d_max = depth_upscaled.amax(dim=[2, 3], keepdim=True)
        norm_depth = (depth_upscaled - d_min) / (d_max - d_min + 1e-8)
        prior = norm_depth.squeeze().cpu().numpy().astype(np.float32)

    np.save(out_path, prior)
    generated_count += 1

    if (generated_count) % 50 == 0:
        print(f"  Processed {generated_count}/{len(target_records)} scenes...")

print(f"Done! Cached {generated_count} new DAV2 priors (already had {cached_count}). Total target: {len(target_records)}")
