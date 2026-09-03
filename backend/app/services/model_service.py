"""
DepthWizard (SIH26175) — Singleton Model Service
Player 4: Backend & Systems Integration Lead

Loads the sealed M2-FINAL RDAH-Net and frozen Depth Anything V2 Small models once
and manages thread-safe GPU inference execution.
"""

import os
import sys
import time
import threading
import hashlib
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.models.rdah_net import RDAHNetCore
from src.models.dav2_wrapper import DepthAnythingV2Wrapper
from backend.app.config import settings

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
                cls._instance.m2_model = None
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

        # 3. SHA-gated M2-FINAL load. There is deliberately no M1 fallback.
        checkpoint_path = settings.MODEL_CHECKPOINT
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"M2-FINAL checkpoint not found: {checkpoint_path}")

        digest = hashlib.sha256()
        with open(checkpoint_path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        if self.checkpoint_sha256.lower() != settings.MODEL_CHECKPOINT_SHA256.lower():
            raise RuntimeError(
                "M2-FINAL checkpoint SHA256 mismatch: "
                f"expected {settings.MODEL_CHECKPOINT_SHA256}, got {self.checkpoint_sha256}"
            )

        print(f"[ModelService] Loading sealed M2-FINAL checkpoint: {checkpoint_path}...")
        self.m2_model = RDAHNetCore(
            d_model=32,
            num_heads=4,
            output_parameterization=settings.MODEL_OUTPUT_PARAMETERIZATION,
        )
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.m2_model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            self.m2_model = None
            raise RuntimeError(f"M2-FINAL checkpoint could not be loaded: {exc}") from exc

        self.m2_model.to(self.device).eval()
        print(f"[ModelService] M2-FINAL verified and loaded ({self.checkpoint_sha256}).")

        # 4. Concurrency Guard Lock for RTX 4060 GPU
        self.gpu_lock = threading.Lock()
        
        self.load_time_s = time.time() - t0
        self._initialized = True
        
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if self.device == "cuda" else 0.0
        print(f"[ModelService] Model Service Initialized in {self.load_time_s:.2f}s (Allocated VRAM: {vram_mb:.1f} MB)")

    def ensure_initialized(self):
        if (
            not getattr(self, '_initialized', False)
            or self.dav2_model is None
            or self.m2_model is None
        ):
            self.initialize()

    def get_health_info(self) -> dict:
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024) if self.device == "cuda" else 0.0
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if self.device == "cuda" else 0.0
        vram_free = vram_total - vram_alloc
        
        return {
            'loaded': self._initialized,
            'name': 'M2-FINAL + Frozen Depth Anything V2 Small',
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
