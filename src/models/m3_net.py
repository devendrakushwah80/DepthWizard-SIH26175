"""
DepthWizard (SIH26175) — M3 Network Architecture
Author: DepthWizard Phase 2 Pipeline

Integrates:
- RDAHNetCore: Dual-Branch MobileViT Encoder (RGB 3-ch + Relative Depth 1-ch)
- CBAM Attention & Cross-Modal Attention
- Transformer contextual reasoning
- Multi-scale PixelShuffle Decoder
- GSD FiLM Conditioning Module (for M3-C):
  Scale-adaptive feature modulation conditioned on metric GSD (gsd_m, gsd_known)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rdah_net import RDAHNetCore


class GSDFiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on ground resolution (GSD).
    Allows network to dynamically scale features for arbitrary sensor GSDs.
    """
    def __init__(self, in_features: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, channels * 2)
        )
        # Initialize gamma to 0, beta to 0 (identity at initialization)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, gsd_vec: torch.Tensor) -> torch.Tensor:
        # gsd_vec: (B, 2) [gsd_m, gsd_known]
        B, C, H, W = feat.shape
        params = self.mlp(gsd_vec)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(B, C, 1, 1) + 1.0  # (1 + gamma) for residual scaling
        beta = beta.view(B, C, 1, 1)
        return gamma * feat + beta


class M3NetCore(nn.Module):
    """
    M3 Metric Height Estimation Network with optional GSD Conditioning.
    """
    def __init__(
        self,
        enable_gsd_conditioning: bool = False,
        d_model: int = 32,
        num_heads: int = 4,
        output_parameterization: str = "unconstrained",
    ):
        super().__init__()
        self.enable_gsd_conditioning = enable_gsd_conditioning
        self.base = RDAHNetCore(
            d_model=d_model,
            num_heads=num_heads,
            output_parameterization=output_parameterization,
        )

        if self.enable_gsd_conditioning:
            # FiLM conditioning at global transformer bottleneck (d_model=32 channels)
            self.film_bottleneck = GSDFiLMBlock(in_features=2, channels=d_model)
            # FiLM conditioning at decoder stage 2 (32 channels)
            self.film_dec2 = GSDFiLMBlock(in_features=2, channels=32)
        else:
            self.film_bottleneck = None
            self.film_dec2 = None

    def forward(
        self,
        rgb: torch.Tensor,
        depth_prior: torch.Tensor,
        gsd_m: Optional[torch.Tensor] = None,
        gsd_known: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.enable_gsd_conditioning or gsd_m is None or gsd_known is None:
            # Pure base forward pass (M3-A / M3-B mode)
            # Note: RDAHNetCore forward takes (depth, img)
            return self.base(depth_prior, rgb)

        # M3-C GSD Conditioned Forward Pass
        # Format GSD vector: (B, 2)
        if gsd_m.ndim == 1:
            gsd_m = gsd_m.unsqueeze(-1)
        if gsd_known.ndim == 1:
            gsd_known = gsd_known.unsqueeze(-1)
        gsd_vec = torch.cat([gsd_m, gsd_known], dim=-1)

        B, _, H_in, W_in = depth_prior.shape

        # 1. Multi-scale feature extraction
        depth_feats = self.base.depth_encoder(depth_prior)
        img_feats = self.base.img_encoder(rgb)

        depth_feats = [self.base.cbam_blocks[i](f) for i, f in enumerate(depth_feats)]
        img_feats = [self.base.cbam_blocks[i](f) for i, f in enumerate(img_feats)]

        # 2. Hierarchical bidirectional cross-modal fusion
        fused_feats = []
        for i in range(3):
            feat1 = self.base.cross_attn_blocks[i](q=depth_feats[i], k=img_feats[i], v=img_feats[i])
            feat2 = self.base.rev_cross_attn_blocks[i](q=img_feats[i], k=depth_feats[i], v=depth_feats[i])
            fused_feats.append((feat1 + feat2) * 0.5)

        # 3. Global contextual transformer
        global_feat = self.base.pos_encoding(fused_feats[2])
        global_feat = self.base.global_transformer(global_feat)

        # Apply GSD FiLM modulation at transformer bottleneck
        global_feat = self.film_bottleneck(global_feat, gsd_vec)

        # 4. Hierarchical PixelShuffle decoding with skip connections
        x = self.base.decoder_stage1(global_feat)
        skip2 = F.interpolate(self.base.skip_projs[2](fused_feats[2]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip2

        x = self.base.decoder_stage2(x)
        # Apply GSD FiLM modulation at intermediate stage
        x = self.film_dec2(x, gsd_vec)
        skip1 = F.interpolate(self.base.skip_projs[1](fused_feats[1]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip1

        x = self.base.decoder_stage3(x)
        skip0 = F.interpolate(self.base.skip_projs[0](fused_feats[0]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip0

        height_pred = self.base.decoder_stage4(x)
        if self.base.output_parameterization == "softplus":
            height_pred = F.softplus(height_pred)

        # Ensure exact input dimensions
        if height_pred.shape[2:] != (H_in, W_in):
            height_pred = F.interpolate(height_pred, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return F.relu(height_pred)
