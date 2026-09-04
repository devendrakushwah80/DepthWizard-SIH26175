"""
DepthWizard (SIH26175) — Singleton Model Service
Player 4: Backend & Systems Integration Lead

Loads the production AGL neural network (M3-FINAL by default, or sealed M2-FINAL)
and frozen Depth Anything V2 Small models once and manages thread-safe GPU inference.
"""

import os
import sys
import time
import threading
import hashlib
from typing import Optional
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.models.rdah_net import RDAHNetCore
from src.models.m3_net import M3NetCore
from src.models.dav2_wrapper import DepthAnythingV2Wrapper
from backend.app.config import settings


class M3AsM2Adapter(torch.nn.Module):
    """
    Adapter enabling M3NetCore to accept legacy (depth, rgb) parameter ordering
    for backward compatibility with existing pipeline callers.
    """
    def __init__(self, m3_core: M3NetCore, device: str):
        super().__init__()
        self.m3_core = m3_core
        self.device = device

    def forward(self, depth_patch: torch.Tensor, rgb_patch: torch.Tensor, gsd_m: Optional[float] = None, **kwargs):
        if gsd_m is not None and float(gsd_m) > 0:
            gsd_val = torch.tensor([[float(gsd_m)]], dtype=torch.float32, device=self.device)
            gsd_known = torch.tensor([[1.0]], dtype=torch.float32, device=self.device)
        else:
            gsd_val = torch.tensor([[0.5]], dtype=torch.float32, device=self.device)
            gsd_known = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)
        return self.m3_core(rgb_patch, depth_patch, gsd_m=gsd_val, gsd_known=gsd_known)


class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelService, cls).__new__(cls)
                cls._instance._initialized = False
                cls._instance.device = "cuda" if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu"
                cls._instance.device_name = torch.cuda.get_device_name(0) if cls._instance.device == "cuda" else "CPU"
                cls._instance.gpu_lock = threading.Lock()
                cls._instance.load_time_s = 0.0
                cls._instance.dav2_model = None
                cls._instance.model = None
                cls._instance.m2_model = None
                cls._instance.model_family = settings.MODEL_NAME
                cls._instance.checkpoint_sha256 = None
            return cls._instance

    def initialize(self):
        if getattr(self, '_initialized', False):
            return
        
        t0 = time.time()
        print("[ModelService] Initializing resident PyTorch models...")
        
        # 1. Device Selection
        if torch.cuda.is_available() and settings.DEVICE == "cuda":
            self.device = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device = "cpu"
            self.device_name = "CPU"
            
        print(f"[ModelService] Selected Compute Device: {self.device} ({self.device_name})")

        # 2. Load and explicitly freeze the locked DAV2 Small prior model.
        print(f"[ModelService] Loading frozen DAV2 Small: {settings.DAV2_MODEL_ID}...")
        self.dav2_model = DepthAnythingV2Wrapper(
            model_id=settings.DAV2_MODEL_ID,
            device=self.device
        )
        for parameter in self.dav2_model.model.parameters():
            parameter.requires_grad = False
        self.dav2_model.model.eval()

        # 3. SHA-gated Model load (M3-FINAL by default, or sealed M2-FINAL).
        checkpoint_path = settings.MODEL_CHECKPOINT
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Model checkpoint not found: {checkpoint_path}")

        digest = hashlib.sha256()
        with open(checkpoint_path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        if self.checkpoint_sha256.lower() != settings.MODEL_CHECKPOINT_SHA256.lower():
            raise RuntimeError(
                f"{settings.MODEL_NAME} checkpoint SHA256 mismatch: "
                f"expected {settings.MODEL_CHECKPOINT_SHA256}, got {self.checkpoint_sha256}"
            )

        print(f"[ModelService] Loading verified {settings.MODEL_NAME} checkpoint: {checkpoint_path}...")
        raw_state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = raw_state.get("model_state_dict", raw_state) if isinstance(raw_state, dict) else raw_state

        if settings.MODEL_NAME == "M3-FINAL":
            self.model_family = "M3-FINAL"
            core_model = M3NetCore(
                enable_gsd_conditioning=True,
                d_model=32,
                num_heads=4,
                output_parameterization=settings.MODEL_OUTPUT_PARAMETERIZATION,
            )
            core_model.load_state_dict(state_dict, strict=True)
            core_model.to(self.device).eval()
            self.model = core_model
            self.m2_model = M3AsM2Adapter(core_model, self.device)
        else:
            self.model_family = "M2-FINAL"
            cleaned_state = {}
            for k, v in state_dict.items():
                if k.startswith("rdah_core."):
                    cleaned_state[k[10:]] = v
                elif k.startswith("base."):
                    cleaned_state[k[5:]] = v
                else:
                    cleaned_state[k] = v
            core_model = RDAHNetCore(
                d_model=32,
                num_heads=4,
                output_parameterization=settings.MODEL_OUTPUT_PARAMETERIZATION,
            )
            core_model.load_state_dict(cleaned_state, strict=True)
            core_model.to(self.device).eval()
            self.model = core_model
            self.m2_model = core_model

        print(f"[ModelService] {self.model_family} verified and loaded ({self.checkpoint_sha256}).")

        # 4. Concurrency Guard Lock for GPU
        self.gpu_lock = threading.Lock()
        
        self.load_time_s = time.time() - t0
        self._initialized = True
        
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if self.device == "cuda" else 0.0
        print(f"[ModelService] Model Service Initialized in {self.load_time_s:.2f}s (Allocated VRAM: {vram_mb:.1f} MB)")

    def predict_patch(
        self,
        depth_patch: torch.Tensor,
        rgb_patch: torch.Tensor,
        gsd_m: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Executes inference on a single crop patch using the active model.
        """
        if self.model_family == "M3-FINAL":
            return self.m2_model(depth_patch, rgb_patch, gsd_m=gsd_m)
        else:
            return self.model(depth_patch, rgb_patch)

    def ensure_initialized(self):
        if (
            not getattr(self, '_initialized', False)
            or self.dav2_model is None
            or self.model is None
        ):
            self.initialize()

    def get_health_info(self) -> dict:
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024) if self.device == "cuda" else 0.0
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if self.device == "cuda" else 0.0
        vram_free = vram_total - vram_alloc
        
        return {
            'loaded': self._initialized,
            'name': f"{self.model_family} + Frozen Depth Anything V2 Small",
            'model_family': self.model_family,
            'device': self.device,
            'checkpoint_path': settings.MODEL_CHECKPOINT,
            'checkpoint_sha256': self.checkpoint_sha256,
            'checkpoint_sha256_verified': self.checkpoint_sha256 == settings.MODEL_CHECKPOINT_SHA256,
            'output_parameterization': settings.MODEL_OUTPUT_PARAMETERIZATION,
            'dav2_model_id': settings.DAV2_MODEL_ID,
            'dav2_frozen': bool(
                self.dav2_model is not None
                and all(not p.requires_grad for p in self.dav2_model.model.parameters())
            ),
            'vram_allocated_mb': round(vram_alloc, 1),
            'gpu_info': {
                'available': self.device == 'cuda',
                'device_name': self.device_name,
                'total_memory_mb': round(vram_total, 1),
                'allocated_memory_mb': round(vram_alloc, 1),
                'free_memory_mb': round(vram_free, 1)
            }
        }

model_service = ModelService()
