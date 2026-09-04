# DepthWizard (SIH26175) — Production Deployment Guide

## 1. System Architecture

```
User Browser
    │
    ▼
Vercel Edge Network
React 19 + Vite 8 + Three.js 0.185 (Frontend)
    │
    │ HTTPS (@gradio/client + REST)
    ▼
Hugging Face Spaces: Devendra80/depthwizard-api
Python 3.10 Inference Engine (Gradio + FastAPI)
    ├── Depth Anything V2 Small (Locked Prior)
    ├── M3-FINAL Checkpoint (GSD FiLM Conditioned AGL Model)
    └── Geospatial / Heatmap / Metadata Processing Pipeline
```

## 2. Verified Production Endpoints

| Component | Target Platform | URL / Identifier | Status |
| :--- | :--- | :--- | :--- |
| **Inference Backend** | Hugging Face Spaces | `https://devendra80-depthwizard-api.hf.space` (`Devendra80/depthwizard-api`) | **LIVE & VERIFIED** |
| **Frontend UI** | Vercel | Configured via `frontend/vercel.json` & `vercel.json` | **READY FOR VERCEL DEPLOY** |

## 3. Checkpoint SHA256 Integrity Verification

Model weights are strictly read-only and verified at application startup:

| Model Checkpoint | File Path | Expected SHA256 Hash | Verification Status |
| :--- | :--- | :--- | :--- |
| **M3-FINAL (Production)** | `models/m3_final/M3_FINAL.pth` | `db1a7646ef087f13284e5806cc8c7b22baf6a8bb23ed9935082db08bbb376330` | **MATCHED (Bitwise Identical)** |
| **M2-FINAL (Frozen)** | `models/m2_final/M2_FINAL.pth` | `6fa4f03dd24726092b75aaf3fa606211c5c66eaaa66ef0dbdbf77eb036bf349f` | **MATCHED (Bitwise Identical)** |

Startup integrity check in `hf_space/app.py`:
```python
computed_hash = hashlib.sha256(Path("models/m3_final/M3_FINAL.pth").read_bytes()).hexdigest()
if computed_hash.lower() != EXPECTED_SHA256.lower():
    raise RuntimeError("M3-FINAL SHA256 mismatch! Halting startup.")
```

## 4. Hardware & Runtime Configuration

- **Compute Platform:** Hugging Face Spaces with ZeroGPU (NVIDIA A100 / A10G dynamic allocation) with CPU fallback.
- **ZeroGPU Allocation:** Decorator `@spaces.GPU(duration=25)` per inference call to minimize quota consumption.
- **CPU Fallback:** On free/unauthenticated requests where ZeroGPU quota is exceeded, automatic fallback to multi-core CPU occurs (~0.51s inference time).
- **Execution Mode:** `torch.inference_mode()`, with `model.eval()` and frozen DAV2 Small parameters (`requires_grad = False`).

## 5. Measured Performance & Latency (Real Remote Benchmark)

*Measurements taken via `@gradio/client` communicating over public HTTPS to `https://devendra80-depthwizard-api.hf.space`:*

| Operation | Measured Duration | Notes |
| :--- | :--- | :--- |
| **Gradio Client Handshake** | 3.76 s | Space wake & API schema sync |
| **`/health` API Query** | < 0.2 s | Returns model name, SHA256 verification, and device |
| **`/predict_agl` (Cold start / with ZeroGPU allocation)** | 6.07 s | Round-trip network + image upload + ZeroGPU allocation + inference |
| **Pure Neural Inference (Remote ZeroGPU)** | **0.074 s** | DAV2 Small: 0.025s, M3-FINAL: 0.039s |
| **Pure Neural Inference (Local CPU fallback)** | **0.510 s** | CPU evaluation without GPU |

## 6. Known Limitations & Operational Guidance

1. **Cold Start Overhead:**
   - If the Space has been idle, Hugging Face may put the container to sleep. The initial request will take 15–30 seconds for the container to wake and load weights into host memory.
   - Subsequent requests take milliseconds for pure model execution.
2. **ZeroGPU Anonymous IP Quota:**
   - Hugging Face applies an hourly rate limit for anonymous unauthenticated ZeroGPU usage.
   - To prevent quota exhaustion, the ZeroGPU function duration is locked to 25s, and CPU fallback handles burst traffic.
3. **Large Image Tiling:**
   - Single-image inference is optimized for images up to 1024×1024. For satellite scenes exceeding 2048×2048, sliding-window tiling is recommended to avoid browser payload timeouts.

## 7. Vercel Deployment Instructions

The repository includes both root `vercel.json` and `frontend/vercel.json`.

### Option A: Via Vercel Web Dashboard (Recommended)
1. Navigate to [vercel.com/new](https://vercel.com/new) and import the repository: `devendrakushwah80/DepthWizard-SIH26175`.
2. Configure Project:
   - **Root Directory:** `frontend`
   - **Framework Preset:** `Vite`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
3. Add Environment Variables:
   - `VITE_DEPLOYMENT_MODE` = `gradio`
   - `VITE_GRADIO_SPACE_ID` = `Devendra80/depthwizard-api`
   - `VITE_API_BASE_URL` = `https://devendra80-depthwizard-api.hf.space`
4. Click **Deploy**.

### Option B: Via Vercel CLI
```bash
npx vercel --cwd frontend
```
Follow the interactive prompts to link your Vercel account.
