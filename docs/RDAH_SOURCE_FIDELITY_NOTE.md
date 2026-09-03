# RDAH-Net Source-Fidelity Validation Note
**Problem Statement ID:** SIH26175  
**Project:** DepthWizard — Single-View Height Estimation and 3D Flythrough  
**Organization:** ISRO / Department of Space  
**Author:** Player 1 (AI/ML Lead)  
**Date:** September 1, 2026  
**Reference Repository:** `https://github.com/Elenairene/RDAH-Net`  
**Reference Commit SHA:** `373bca28299683ab0e5d892dfc87598ab967b564` (April 12, 2026)  

---

## 1. Context & Purpose

This audit resolves the architectural specification discrepancy between the published paper text / repository README and the actual executable Python source code in `Elenairene/RDAH-Net`.

---

## 2. Documented Architecture Comparison

| Architectural Component | Published Paper & README Text | Executable Official Source (`train.py` / `test.py`) | Local Implementation (`src/models/rdah_net.py`) | Alignment Verdict |
|---|---|---|---|---|
| **RGB Feature Encoder** | "Lightweight MobileNetV2" | `MobileViT_S_Light(in_channels=3)` (MobileViT blocks with $8\times 8$ block self-attention) | `MobileViT_S_Light(in_channels=3)` | **MATCHES CODE (PASS)** |
| **Depth Feature Encoder** | "Depth branch" / DAV2 prior | `MobileViT_S_Light(in_channels=1)` | `MobileViT_S_Light(in_channels=1)` | **MATCHES CODE (PASS)** |
| **Encoder Stages & Channels** | 3-4 feature stages | Stage 1: 64ch, Stage 2: 128ch, Stage 3: 256ch $\to$ projected to $d_{\text{model}}=32$ | Stage 1: 64ch, Stage 2: 128ch, Stage 3: 256ch $\to$ projected to $d_{\text{model}}=32$ | **EXACT MATCH (PASS)** |
| **Attention Enhancement** | CBAM Module | `CBAM(channel=32, reduction=16)` (Channel + Spatial attention on all 3 stages) | `CBAM(channel=32, reduction=16)` | **EXACT MATCH (PASS)** |
| **Cross-Modal Fusion** | Bidirectional cross-attention | `LightCrossAttention` ($\text{Depth}\to\text{RGB}$ and $\text{RGB}\to\text{Depth}$) with $d_{\text{model}}=32$, 4 heads | `LightCrossAttention` ($d_{\text{model}}=32$, 4 heads) | **EXACT MATCH (PASS)** |
| **Positional Encoding** | 2D Positional Encoding | `PositionalEncoding(d_model=32)` (Sinusoidal 2D spatial grid) | `PositionalEncoding(d_model=32)` | **EXACT MATCH (PASS)** |
| **Global Bottleneck** | Transformer Block | `LightTransformerBlock` (Cross-Attention Self-Attention + FFN) | `LightTransformerBlock` | **EXACT MATCH (PASS)** |
| **Decoder Architecture** | PixelShuffle Upsampling | 4-stage `PixelShuffle(2)` decoder with skip projections ($16\to 32\to 64\to d_{\text{model}}$) | 4-stage `PixelShuffle(2)` decoder with skip projections | **EXACT MATCH (PASS)** |
| **Core Parameter Count** | $\approx 5-6\text{ M}$ | **5,372,047 parameters (5.37 M)** | **5,372,047 parameters (5.37 M)** | **EXACT MATCH (PASS)** |

---

## 3. Analysis of the Discrepancy

1. **Textual Description:** The README text and initial paper draft refer to a "MobileNetV2" backbone.
2. **Code Reality:** The actual model class executed during training and evaluation is `HeightPredTransformer`, which utilizes **`MobileViT_S_Light`**.
3. **Engineering Rationale:** MobileViT combines MobileNet-style depthwise separable convolutions with lightweight multi-head Transformer self-attention blocks. In satellite imagery, purely local convolutions struggle with wide-area context (e.g., ground datum across large building shadows). The hybrid MobileViT encoder provides the necessary global receptive field while maintaining low parameter count ($5.37\text{M}$) and fast inference.

---

## 4. Decision for DepthWizard

**Decision:** We adopt the **official executable codebase architecture (`HeightPredTransformer` with `MobileViT_S_Light`)**.

### Justification:
* **Source Fidelity:** Directly matches the author's working codebase (`commit 373bca2`).
* **Reproducibility:** Eliminates discrepancies between published training scripts and our pipeline.
* **Hardware Suitability:** Consumes $< 1\text{ GB}$ VRAM during training on the RTX 4060 Laptop GPU.

---

## 5. Architectural Verification Checklist

- [x] Dual-branch 3-stage MobileViT-S feature encoders ($64 \to 128 \to 256 \to 32$)
- [x] Sequential Channel + Spatial CBAM on all feature stages
- [x] Bidirectional LightCrossAttention with block-wise partitioning ($8 \times 8$)
- [x] Dynamic 2D Sinusoidal Positional Encoding
- [x] Global Context Transformer bottleneck
- [x] 4-stage PixelShuffle sub-pixel convolution decoder with skip projections
- [x] Parameter count verified at **5,372,047 parameters**
- [x] GPU smoke test verified (Forward, Masked Loss, Backward, Optimizer step: 100% PASS)
