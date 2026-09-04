"""
DepthWizard (SIH26175) — Fast Reliable System Launcher & Preflight
ISRO / Department of Space — Smart India Hackathon 2026

Starts Backend (FastAPI on :8000) and Frontend (Vite on :3000), verifies health, and opens the web application.
"""

import os
import sys
import time
import socket
import subprocess
import urllib.request
import json
import webbrowser
import hashlib
import functools

print = functools.partial(print, flush=True)

M2_EXPECTED_SHA256 = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"
M3_EXPECTED_SHA256 = "db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330"
EXPECTED_M2_SHA256 = M2_EXPECTED_SHA256
EXPECTED_M3_SHA256 = M3_EXPECTED_SHA256
DAV2_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

def verify_sha256(file_path, expected_sha: str) -> bool:
    if not os.path.exists(file_path):
        return False
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_sha.lower()

def is_port_in_use(port):
    for host in ('localhost', '127.0.0.1'):
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except Exception:
            pass
    return False

def check_frontend_ready():
    for url in ("http://localhost:3000", "http://127.0.0.1:3000"):
        try:
            req = urllib.request.urlopen(url, timeout=1)
            if req.getcode() == 200:
                return True
        except Exception:
            pass
    return False

def check_backend_health():
    for url in ("http://127.0.0.1:8000/health", "http://localhost:8000/health"):
        try:
            req = urllib.request.urlopen(url, timeout=2)
            data = json.loads(req.read().decode())
            if data.get("status") == "ok":
                return True, data
        except Exception:
            pass
    return False, None

def run_preflight_checks(project_root: str):
    print("=" * 80)
    print("  ISRO DEPTHWIZARD (SIH26175) — STARTUP PREFLIGHT CHECKLIST")
    print("=" * 80)

    # 1. M2 Checkpoint Present & SHA
    m2_ckpt = os.path.join(project_root, "models", "m2_final", "M2_FINAL.pth")
    m2_present = os.path.exists(m2_ckpt)
    m2_sha_ok = False
    if m2_present:
        hasher = hashlib.sha256()
        with open(m2_ckpt, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        m2_sha_ok = (hasher.hexdigest().lower() == M2_EXPECTED_SHA256)

    # 2. M3 Checkpoint Present & SHA
    m3_ckpt = os.path.join(project_root, "models", "m3_final", "M3_FINAL.pth")
    m3_present = os.path.exists(m3_ckpt)
    m3_sha_ok = False
    if m3_present:
        hasher = hashlib.sha256()
        with open(m3_ckpt, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        m3_sha_ok = (hasher.hexdigest().lower() == M3_EXPECTED_SHA256)

    # 3. DAV2 Available / Cached
    dav2_cached = False
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(DAV2_MODEL_ID, "config.json")
        dav2_cached = isinstance(res, str)
    except Exception:
        # Fallback check under ~/.cache/huggingface/hub
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf")
        dav2_cached = os.path.exists(cache_dir)

    # 4. CUDA Available & Selected Device
    cuda_available = False
    selected_device = "cpu"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            selected_device = f"cuda ({torch.cuda.get_device_name(0)})"
        else:
            selected_device = "cpu"
    except Exception:
        pass

    # 5. Outputs Directory Writable
    outputs_dir = os.path.join(project_root, "outputs")
    outputs_writable = False
    try:
        os.makedirs(outputs_dir, exist_ok=True)
        test_file = os.path.join(outputs_dir, ".preflight_write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        outputs_writable = True
    except Exception:
        outputs_writable = False

    # 6. Runtime Configuration Valid
    storage_dir = os.path.join(project_root, "storage", "scenes")
    os.makedirs(storage_dir, exist_ok=True)
    runtime_config_valid = (m3_present and m3_sha_ok) or (m2_present and m2_sha_ok) and outputs_writable

    # Report Preflight Checklist
    print(f"  [1] Active Production Model:            {'M3-FINAL (FiLM GSD-Conditioned)' if m3_present else 'M2-FINAL'}")
    print(f"  [2] M3 checkpoint present / SHA:        {'YES / VERIFIED' if (m3_present and m3_sha_ok) else 'NO'}")
    print(f"  [3] M2 checkpoint present / SHA:        {'YES / VERIFIED' if (m2_present and m2_sha_ok) else 'NO'}")
    print(f"  [4] DAV2 available/cached:              {'YES' if dav2_cached else 'NO'}")
    print(f"  [5] CUDA available:                     {'YES' if cuda_available else 'NO'}")
    print(f"  [6] Selected inference device:          {selected_device}")
    print(f"  [7] Outputs directory writable:         {'YES' if outputs_writable else 'NO'}")
    print(f"  [8] Required runtime config valid:      {'YES' if runtime_config_valid else 'NO'}")
    print("=" * 80)

    if not dav2_cached:
        print("  NOTE: If DAV2 is not cached, download it in advance using:")
        print("    python -c \"from transformers import AutoModelForDepthEstimation; AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')\"")
        print("=" * 80)

    if not (m3_present and m3_sha_ok) and not (m2_present and m2_sha_ok):
        print(f"[FATAL ERROR] Model checkpoint integrity failed. Exiting.")
        sys.exit(1)

    return runtime_config_valid

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Execute Preflight Checklist
    run_preflight_checks(project_root)

    python_exe = sys.executable
    backend_proc = None

    logs_dir = os.path.join(project_root, "outputs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    backend_log_file = os.path.join(logs_dir, "backend_startup.log")
    frontend_log_file = os.path.join(logs_dir, "frontend_startup.log")

    # Set active production environment defaults
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass

    os.environ.setdefault("DEPTHWIZARD_MODEL_FAMILY", "M3-FINAL")
    os.environ.setdefault("DEPTHWIZARD_MODEL_CHECKPOINT", os.path.join(project_root, "models", "m3_final", "M3_FINAL.pth"))
    os.environ.setdefault("DEPTHWIZARD_DEVICE", "cuda" if has_cuda else "cpu")

    # 1. Check / Start Backend (:8000)
    print("\n[Step 1] Checking FastAPI Backend on http://127.0.0.1:8000...")
    is_healthy, h_data = check_backend_health()
    if is_healthy:
        print(f"  [OK] Backend is already running! (Model: {h_data['model']['name']} on {h_data['model']['device']})")
    else:
        print(f"  Starting FastAPI Backend Server on port 8000 (logging to {backend_log_file})...")
        backend_log = open(backend_log_file, "w", encoding="utf-8")
        backend_proc = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=project_root,
            stdout=backend_log,
            stderr=subprocess.STDOUT
        )
        for _ in range(35):
            time.sleep(1)
            is_healthy, h_data = check_backend_health()
            if is_healthy:
                print(f"  [OK] Backend Online! Resident Model: {h_data['model']['name']} on {h_data['model']['device']}")
                break
        if not is_healthy:
            print("\n  [FATAL ERROR] Backend failed to start on port 8000!")
            print("  --- BACKEND STARTUP LOG (Last 25 lines) ---")
            if os.path.exists(backend_log_file):
                with open(backend_log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print("".join(lines[-25:]))
            print("  -------------------------------------------")
            sys.exit(1)

    # 2. Check / Start Frontend (:3000)
    print("\n[Step 2] Checking Vite React Frontend on http://localhost:3000...")
    frontend_proc = None
    frontend_in_use = is_port_in_use(3000) or check_frontend_ready()
    if frontend_in_use:
        print("  [OK] Frontend is already running on port 3000!")
    else:
        print(f"  Starting Vite React Frontend Server (logging to {frontend_log_file})...")
        frontend_dir = os.path.join(project_root, "frontend")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        frontend_log = open(frontend_log_file, "w", encoding="utf-8")
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            shell=(sys.platform == "win32"),
            stdout=frontend_log,
            stderr=subprocess.STDOUT
        )
        for _ in range(25):
            time.sleep(1)
            if is_port_in_use(3000) or check_frontend_ready():
                print("  [OK] Frontend Dev Server started!")
                break
        if not (is_port_in_use(3000) or check_frontend_ready()):
            print("\n  [FATAL ERROR] Frontend failed to start on port 3000!")
            print("  --- FRONTEND STARTUP LOG (Last 25 lines) ---")
            if os.path.exists(frontend_log_file):
                with open(frontend_log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print("".join(lines[-25:]))
            print("  --------------------------------------------")
            sys.exit(1)

    # 3. Validation or Interactive Loop
    print("\n" + "=" * 80)
    print("  DEPTHWIZARD IS ONLINE AND READY FOR SIH EVALUATION!")
    print("    -> Web Application URL: http://localhost:3000/")
    print("    -> Backend API Docs:    http://localhost:8000/docs")
    print("    -> Backend Health API:  http://localhost:8000/health")
    print("=" * 80)

    if "--validate" in sys.argv:
        print("Running automatic validation check...")
        h_ok, h_info = check_backend_health()
        assert h_ok, "Backend health check failed!"
        print(f"  [Validation] Backend Health OK: {h_info}")
        f_in_use = is_port_in_use(3000) or check_frontend_ready()
        assert f_in_use, "Frontend port 3000 not responding!"
        print(f"  [Validation] Frontend Port 3000 Responding OK")
        print("  [Validation] Standalone Launch Validation: ALL CHECKS PASSED")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                frontend_proc.terminate()
        sys.exit(0)

    if not os.environ.get("CI") and not os.environ.get("NO_BROWSER"):
        print("Opening web browser at http://localhost:3000/ ...")
        webbrowser.open("http://localhost:3000/")

    try:
        print("\nDepthWizard is running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down DepthWizard...")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                frontend_proc.terminate()

if __name__ == '__main__':
    main()
