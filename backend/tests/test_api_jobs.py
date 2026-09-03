"""
DepthWizard (SIH26175) — Job Processing & Validation Tests
Player 4: Backend & Systems Integration Lead
"""

import io
import time
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_job_submission_and_status():
    # Create test 512x512 RGB image in-memory
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 1. Submit Job
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("test_unit.png", buf, "image/png")},
        data={"generate_3d": "false", "generate_pointcloud": "false"}
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]
    assert data["status"] in ["queued", "validating", "running_inference", "completed"]

    # 2. Query Status
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ["queued", "validating", "running_inference", "completed"]

def test_invalid_file_extension():
    buf = io.BytesIO(b"dummy binary executable")
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("malicious.exe", buf, "application/octet-stream")}
    )
    assert response.status_code == 415
    data = response.json()
    assert data["error"]["code"] == "INVALID_FILE_TYPE"

def test_nonexistent_job():
    response = client.get("/api/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "JOB_NOT_FOUND"
