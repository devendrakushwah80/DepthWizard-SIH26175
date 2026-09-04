"""
Test real end-to-end backend inference with promoted M3-FINAL production weights.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.services.model_service import model_service
from backend.app.services.inference_service import inference_service


def main():
    print("=" * 70)
    print("DepthWizard SIH26175 — Real M3-FINAL Backend End-to-End Test")
    print("=" * 70)
    print(f"Configured Model: {settings.MODEL_NAME}")
    print(f"Configured Checkpoint: {settings.MODEL_CHECKPOINT}")
    print(f"Expected SHA256: {settings.MODEL_CHECKPOINT_SHA256}")

    model_service.initialize()
    health = model_service.get_health_info()
    print("\nModel Health Status:")
    for k, v in health.items():
        if k != "gpu_info":
            print(f"  {k}: {v}")

    assert health["loaded"] is True
    assert health["checkpoint_sha256_verified"] is True

    # Run inference on synthetic test image (512x512x3)
    np.random.seed(42)
    test_rgb = np.random.randint(50, 200, size=(512, 512, 3), dtype=np.uint8)

    print("\nExecuting real full-resolution inference with M3...")
    result = inference_service.predict_height_map(test_rgb, gsd_m=0.3)
    pred_agl = result["predicted_agl_m"]
    dav2 = result["dav2_prior"]
    rel_surf = result["relative_surface"]

    print(f"Inference Time: {result['total_inference_time_s']:.3f}s")
    print(f"Predicted AGL shape: {pred_agl.shape}, min: {pred_agl.min():.2f}m, max: {pred_agl.max():.2f}m, mean: {pred_agl.mean():.2f}m")
    print(f"Relative surface shape: {rel_surf.shape}, min: {rel_surf.min():.2f}, max: {rel_surf.max():.2f}")
    assert np.isfinite(pred_agl).all()
    assert (pred_agl >= 0).all()
    print("\n[SUCCESS] M3-FINAL Real Backend Pipeline Verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
