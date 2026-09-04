# DepthWizard (SIH26175) — Google Cloud Run Production Deployment

## Architecture Overview

```
User Web Browser
      │
      ▼ (HTTPS)
Vercel Edge Network
React 19 + Vite + Three.js 3D Workspace
(https://depth-wizard-sih-26175.vercel.app)
      │
      ▼ (HTTPS REST)
Google Cloud Run (asia-south1)
FastAPI Backend (M3-FINAL + DAV2 Small)
[2 vCPU, 4 GiB RAM, Gen2, Concurrency=1, Timeout=300s]
      │
      ▼ (FUSE Volume Mount)
Google Cloud Storage Bucket (gs://${PROJECT_ID}-depthwizard-scenes)
Mounted at /mnt/depthwizard:
  ├── /mnt/depthwizard/scenes/   (Persistent Scene Artifacts: GLB, PLY, NPY)
  └── /mnt/depthwizard/uploads/  (Temporary Image Upload Ingestion)
```

## Prerequisites

1. Google Cloud Project with Billing Enabled.
2. `gcloud` CLI installed or Google Cloud Shell.
3. Authenticated session:
   ```bash
   gcloud auth login
   gcloud config set project <YOUR_PROJECT_ID>
   ```

## Automated 1-Command Deployment

Execute the deployment script from repository root:

```bash
chmod +x deploy/gcp/deploy.sh
./deploy/gcp/deploy.sh
```

Or run via Google Cloud Shell:
```bash
gcloud builds submit --config deploy/gcp/cloudbuild.yaml --substitutions="_IMAGE_TAG=asia-south1-docker.pkg.dev/$(gcloud config get-value project)/depthwizard-repo/depthwizard-api:latest" .
```

## Cloud Run Service Specification

- **Compute:** 2 vCPU, 4 GiB Memory (CPU-first inference)
- **Container Concurrency:** 1 (dedicated single-job processing per container instance)
- **Scaling Limits:** Min instances: 0, Max instances: 2
- **Execution Environment:** Second generation (`gen2` required for Cloud Storage volume mounts)
- **Startup CPU Boost:** Enabled
- **Request Timeout:** 300 seconds
- **Environment Variables:**
  - `DEPTHWIZARD_DEVICE=cpu`
  - `DEPTHWIZARD_MODEL_FAMILY=M3-FINAL`
  - `DEPTHWIZARD_STORAGE_DIR=/mnt/depthwizard/scenes`
  - `DEPTHWIZARD_UPLOAD_TEMP_DIR=/mnt/depthwizard/uploads`
  - `DEPTHWIZARD_CORS_ORIGINS=https://depth-wizard-sih-26175.vercel.app`

## Vercel Frontend Configuration

Once Cloud Run outputs your service URL (e.g. `https://depthwizard-api-xyz-as.a.run.app`), update Vercel environment variables:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `https://depthwizard-api-<hash>-<region>.a.run.app` | Cloud Run FastAPI REST endpoint |
| `VITE_DEPLOYMENT_MODE` | `workspace` | Enables full 3D Workspace by default |
| `VITE_GRADIO_SPACE_ID` | `Devendra80/depthwizard-api` | Secondary experimental ZeroGPU estimator |

## Health & Model Check

Verify the deployment remotely:
```bash
curl -s https://<YOUR_CLOUD_RUN_URL>/health | jq .
```
Expected output:
```json
{
  "status": "healthy",
  "project": "DepthWizard SIH26175 API",
  "version": "2.0.0",
  "device": "cpu",
  "model": {
    "name": "M3-FINAL",
    "checkpoint_sha256": "db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330",
    "sha256_verified": true
  }
}
```
