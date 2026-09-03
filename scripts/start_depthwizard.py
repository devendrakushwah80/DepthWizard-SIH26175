"""
DepthWizard (SIH26175) — Fast Reliable System Launcher
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

M2_SHA256 = "6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def check_backend_health():
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        data = json.loads(req.read().decode())
        return data.get("status") == "ok", data
    except Exception:
        return False, None

def main():
    print("=" * 80)
    print("  ISRO DEPTHWIZARD (SIH26175)")
    print("  Single-View Height Estimation and 3D Flythrough")
    print("=" * 80)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # 1. Check Python Virtual Environment
    python_exe = sys.executable
    print(f"[Step 1] Python Runtime: {python_exe}")

    # 2. Check Model Checkpoint
    ckpt_path = os.path.join(project_root, "models", "m2_final", "M2_FINAL.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Error] M2-FINAL checkpoint not found at: {ckpt_path}")
        sys.exit(1)
    digest = hashlib.sha256()
    with open(ckpt_path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != M2_SHA256:
        print(f"[Error] M2-FINAL SHA256 mismatch: {digest.hexdigest()}")
        sys.exit(1)
    print(f"[Step 2] Verified sealed M2-FINAL ({os.path.getsize(ckpt_path)/(1024*1024):.2f} MB, SHA256 OK)")

    # 3. Check / Start Backend (:8000)
    print("[Step 3] Checking FastAPI Backend on http://127.0.0.1:8000...")
    is_healthy, h_data = check_backend_health()
    if is_healthy:
        print(f"[OK] Backend is already running and ready! (Model: {h_data['model']['name']})")
    else:
        print("Starting FastAPI Backend Server on port 8000...")
        backend_proc = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Wait up to 30 seconds for model preloading
        for _ in range(30):
            time.sleep(1)
            is_healthy, h_data = check_backend_health()
            if is_healthy:
                print(f"[OK] Backend Online! Resident Model: {h_data['model']['name']} on {h_data['model']['device']}")
                break
        if not is_healthy:
            print("[Error] Backend failed to start. Please check if another process is using port 8000.")
            sys.exit(1)

    # 4. Check / Start Frontend (:3000)
    print("[Step 4] Checking Vite React Frontend on http://127.0.0.1:3000...")
    frontend_in_use = is_port_in_use(3000)
    if frontend_in_use:
        print("[OK] Frontend is already running on port 3000!")
    else:
        print("Starting Vite React Frontend Server...")
        frontend_dir = os.path.join(project_root, "frontend")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        print("[OK] Frontend Dev Server started!")

    # 5. Open Web Browser
    print("=" * 80)
    print("DEPTHWIZARD IS ONLINE AND READY FOR SIH EVALUATION!")
    print("  -> Web Application URL: http://localhost:3000/")
    print("  -> Backend API Docs:    http://localhost:8000/docs")
    print("  -> Backend Health API:  http://localhost:8000/health")
    print("=" * 80)
    print("Opening web browser at http://localhost:3000/ ...")
    webbrowser.open("http://localhost:3000/")

if __name__ == '__main__':
    main()
