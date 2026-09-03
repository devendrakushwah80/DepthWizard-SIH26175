"""
DepthWizard (SIH26175) — CUDA Environment & Smoke Test Script
Player 1: AI/ML Lead
"""

import sys
import os
import time

def check_cuda_env():
    print("=" * 70)
    print("CUDA & PYTORCH ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print("Python Executable:", sys.executable)
    print("Python Version:", sys.version)

    try:
        import torch
        print("\n--- PyTorch Status ---")
        print("PyTorch Version:", torch.__version__)
        print("CUDA Available:", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            dev_idx = 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            cap = torch.cuda.get_device_capability(dev_idx)
            vram_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
            vram_gb = vram_bytes / (1024**3)
            
            print(f"Device [{dev_idx}]: {dev_name}")
            print(f"Compute Capability: {cap[0]}.{cap[1]}")
            print(f"Total VRAM: {vram_gb:.2f} GB ({vram_bytes:,} bytes)")
            print(f"CUDA Version (Torch Build): {torch.version.cuda}")
            if hasattr(torch.backends, 'cudnn'):
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
                print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
                
            # Perform CUDA Tensor Operations Smoke Test
            print("\n--- Running CUDA Tensor Smoke Test ---")
            t0 = time.time()
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - t0) * 1000.0
            
            alloc_mb = torch.cuda.memory_allocated(dev_idx) / (1024**2)
            res_mb = torch.cuda.memory_reserved(dev_idx) / (1024**2)
            
            print(f"Matrix Multiplication (2048x2048) on CUDA: SUCCESS ({elapsed_ms:.2f} ms)")
            print(f"Allocated VRAM: {alloc_mb:.2f} MB | Reserved VRAM: {res_mb:.2f} MB")
            print("\nCUDA SMOKE TEST PASSED: GPU is fully active and ready for neural training!")
        else:
            print("\nWARNING: torch.cuda.is_available() is False. Running on CPU.")
            
    except ImportError as e:
        print("Error importing PyTorch:", e)

    print("=" * 70)

if __name__ == '__main__':
    check_cuda_env()
