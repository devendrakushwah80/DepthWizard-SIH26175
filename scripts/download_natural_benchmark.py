"""
DepthWizard (SIH26175) — Autonomous Natural Domain Benchmark Builder
Acquires real high-resolution NAIP optical imagery and USGS 3DEP 1m bare-earth elevation (DEM)
across 10 geographically diverse natural, forest, and mountainous environments in the United States.
"""

from __future__ import annotations

import os
import json
import time
import urllib.request
from pathlib import Path

OUT_DIR = Path("data/m3/raw/natural")
MANIFEST_PATH = Path("data/m3/manifests/natural_domain_manifest.json")

# 10 Diverse Geographic AOIs (EPSG:4326 bounding boxes: min_lon, min_lat, max_lon, max_lat)
AOIS = [
    {
        "id": "SMOKY_01_FOREST",
        "name": "Great Smoky Mountains Dense Forest",
        "region": "Tennessee/North Carolina",
        "category": "dense_forest",
        "bbox": "-83.52,35.65,-83.50,35.67",
        "expected_features": "Closed deciduous/evergreen canopy, steep ridges"
    },
    {
        "id": "SMOKY_02_RIDGE",
        "name": "Smoky Mountains High Ridge",
        "region": "North Carolina",
        "category": "dense_forest",
        "bbox": "-83.45,35.60,-83.43,35.62",
        "expected_features": "Steep slopes, dense mountain forest"
    },
    {
        "id": "ROCKIES_01_ALPINE",
        "name": "Colorado Rockies Alpine Slope",
        "region": "Colorado",
        "category": "mountainous_steep",
        "bbox": "-105.65,39.75,-105.63,39.77",
        "expected_features": "Subalpine conifers, steep rock faces, high relief (>800m)"
    },
    {
        "id": "ROCKIES_02_VALLEY",
        "name": "Colorado Rockies Mountain Valley",
        "region": "Colorado",
        "category": "mountainous_steep",
        "bbox": "-105.70,39.80,-105.68,39.82",
        "expected_features": "Steep valley walls, mixed pine and aspen"
    },
    {
        "id": "CASCADES_01_RAINFOREST",
        "name": "Pacific Northwest Rainforest",
        "region": "Washington",
        "category": "dense_forest",
        "bbox": "-121.75,46.85,-121.73,46.87",
        "expected_features": "Dense temperate rainforest, tall Douglas fir (>40m)"
    },
    {
        "id": "SIERRA_01_FOOTHILLS",
        "name": "Sierra Nevada Hilly Foothills",
        "region": "California",
        "category": "hilly_mixed",
        "bbox": "-120.35,38.75,-120.33,38.77",
        "expected_features": "Mixed oak woodland, rolling topography"
    },
    {
        "id": "APPALACHIAN_01_CANOPY",
        "name": "Shenandoah Valley Mountain Flank",
        "region": "Virginia",
        "category": "hilly_mixed",
        "bbox": "-78.45,38.55,-78.43,38.57",
        "expected_features": "Hardwood forest on moderate-to-steep slopes"
    },
    {
        "id": "OZARKS_01_HOLLOW",
        "name": "Ozark Plateau Hilly Hollow",
        "region": "Missouri/Arkansas",
        "category": "hilly_mixed",
        "bbox": "-93.25,36.50,-93.23,36.52",
        "expected_features": "Dissected plateau, dense deciduous forest"
    },
    {
        "id": "WHITE_MTNS_01_SLOPE",
        "name": "White Mountains Northern Hardwoods",
        "region": "New Hampshire",
        "category": "mountainous_steep",
        "bbox": "-71.35,44.25,-71.33,44.27",
        "expected_features": "Boreal-transition forest, high relief"
    },
    {
        "id": "BLACK_HILLS_01_PINE",
        "name": "Black Hills Ponderosa Forest",
        "region": "South Dakota",
        "category": "dense_forest",
        "bbox": "-103.55,43.85,-103.53,43.87",
        "expected_features": "Dense ponderosa pine forest on rugged granite hills"
    }
]


def fetch_aoi(aoi: dict) -> dict:
    aoi_id = aoi["id"]
    category = aoi["category"]
    bbox = aoi["bbox"]
    
    sub_dir = OUT_DIR / category
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    naip_path = sub_dir / f"{aoi_id}_naip_rgb.tif"
    dem_path = sub_dir / f"{aoi_id}_dem_dtm.tif"
    
    # 1. NAIP optical (1024x1024)
    naip_url = (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
        f"?bbox={bbox}&bboxSR=4326&size=1024,1024&imageSR=4326&format=tiff&f=image"
    )
    if not (naip_path.exists() and naip_path.stat().st_size > 100_000):
        print(f"Downloading NAIP for {aoi_id}...")
        urllib.request.urlretrieve(naip_url, naip_path)
        time.sleep(0.5)

    # 2. 3DEP elevation DEM (1024x1024)
    dem_url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
        f"?bbox={bbox}&bboxSR=4326&size=1024,1024&imageSR=4326&format=tiff&f=image"
    )
    if not (dem_path.exists() and dem_path.stat().st_size > 100_000):
        print(f"Downloading 3DEP DEM for {aoi_id}...")
        urllib.request.urlretrieve(dem_url, dem_path)
        time.sleep(0.5)

    return {
        "aoi_id": aoi_id,
        "name": aoi["name"],
        "region": aoi["region"],
        "category": aoi["category"],
        "bbox": bbox,
        "naip_path": str(naip_path),
        "dem_path": str(dem_path),
        "naip_size": naip_path.stat().st_size if naip_path.exists() else 0,
        "dem_size": dem_path.stat().st_size if dem_path.exists() else 0,
        "status": "VALID" if (naip_path.exists() and dem_path.exists()) else "ERROR"
    }


def main():
    print("=" * 70)
    print("DepthWizard SIH26175 — Natural Domain Benchmark Builder")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for aoi in AOIS:
        try:
            res = fetch_aoi(aoi)
            print(f"[{res['status']}] {aoi['id']}: {aoi['name']} ({res['category']})")
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {aoi['id']}: {e}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    valid_count = sum(1 for r in results if r["status"] == "VALID")
    print("=" * 70)
    print(f"Natural Domain Benchmark Acquisition Complete: {valid_count}/{len(AOIS)} AOIs ready.")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
