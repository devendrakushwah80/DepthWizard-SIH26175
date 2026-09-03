"""
DepthWizard (SIH26175) — Official RDAH-Net Architecture & GAMUS Integration Wrapper
Player 1: AI/ML Lead

Reference:
- Paper: 'RDAH-Net: Bridging Relative Depth and Absolute Height for Monocular Height Estimation in Remote Sensing' (Remote Sensing, 2026)
- Official Code: https://github.com/Elenairene/RDAH-Net

Components:
1. MobileViT_S_Light: Dual-branch lightweight encoder (RGB 3-ch + Relative Depth 1-ch)
2. CBAM: Channel & Spatial Attention on multi-scale features
3. LightCrossAttention: Bidirectional Cross-Modal Attention (RGB <-> Depth)
4. PositionalEncoding + LightTransformerBlock: Global contextual transformer
5. Multi-Scale PixelShuffle Decoder: Sub-pixel convolution upsampling with skip connections
6. RDAHNetDepthWizard: End-to-end wrapper combining frozen Depth Anything V2 + RDAH-Net
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleConv(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.upsample(x))

class ConvLayer(nn.Module):
    """MobileViT basic convolution block (Conv + BN + Activation)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class BlockAttention(nn.Module):
    """Block-wise Self-Attention with multi-head projection"""
    def __init__(self, dim, num_heads=4, block_size=8, mlp_dim=None, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.mlp_dim = mlp_dim or dim * 2
        self.scale = self.head_dim ** -0.5

        self.local_proj = ConvLayer(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.mlp_dim, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.local_proj(x)
        
        num_blocks_h = max(1, H // self.block_size)
        num_blocks_w = max(1, W // self.block_size)
        num_blocks = num_blocks_h * num_blocks_w
        
        # Reshape to block structure
        # Pad dynamically if H or W not divisible by block_size
        pad_h = (self.block_size - (H % self.block_size)) % self.block_size
        pad_w = (self.block_size - (W % self.block_size)) % self.block_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_p, W_p = x.shape
            num_blocks_h = H_p // self.block_size
            num_blocks_w = W_p // self.block_size
            num_blocks = num_blocks_h * num_blocks_w

        x_blocked = x.reshape(
            B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size
        ).permute(0, 2, 4, 1, 3, 5).reshape(B * num_blocks, C, self.block_size, self.block_size)

        B_blocked, C_blocked, h, w = x_blocked.shape
        n = h * w
        qkv = self.qkv(x_blocked)
        qkv = qkv.reshape(B_blocked, 3, C_blocked, n).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(B_blocked, self.num_heads, self.head_dim, n)
        k = k.reshape(B_blocked, self.num_heads, self.head_dim, n)
        v = v.reshape(B_blocked, self.num_heads, self.head_dim, n)

        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)
        out = out.contiguous().reshape(B_blocked, C_blocked, h, w)
        out = self.proj_drop(self.proj(out))

        out = out.reshape(
            B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(B, C, num_blocks_h * self.block_size, num_blocks_w * self.block_size)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        x = x[:, :, :H, :W] + local_feat + out
        x = x + self.mlp(x)
        return x

class MobileViTBlock(nn.Module):
    """MobileViT block with local conv + block self-attention"""
    def __init__(self, in_channels, out_channels, stride=1, num_heads=4, block_size=8, dropout=0.0):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        assert out_channels % num_heads == 0, f"out_channels={out_channels} must be divisible by num_heads={num_heads}"
        self.attention = BlockAttention(out_channels, num_heads, block_size, dropout=dropout)
        self.conv2 = ConvLayer(out_channels, out_channels, groups=out_channels)

    def forward(self, x):
        return self.conv2(self.attention(self.conv1(x)))

class MobileViT_S_Light(nn.Module):
    """Lightweight MobileViT-S 3-stage feature extractor"""
    def __init__(self, in_channels=3, d_model=32):
        super().__init__()
        self.in_channels = in_channels
        self.stem = ConvLayer(in_channels, 32, kernel_size=4, stride=2, padding=1)

        self.stage1 = nn.Sequential(
            MobileViTBlock(32, 64, stride=2, num_heads=4, block_size=8),
            MobileViTBlock(64, 64, stride=1, num_heads=4, block_size=8)
        )
        self.stage2 = nn.Sequential(
            MobileViTBlock(64, 128, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(128, 128, stride=1, num_heads=8, block_size=8)
        )
        self.stage3 = nn.Sequential(
            MobileViTBlock(128, 256, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(256, 256, stride=1, num_heads=8, block_size=8)
        )

        self.proj1 = ConvLayer(64, d_model, kernel_size=1, padding=0)
        self.proj2 = ConvLayer(128, d_model, kernel_size=1, padding=0)
        self.proj3 = ConvLayer(256, d_model, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)

        return [self.proj1(feat1), self.proj2(feat2), self.proj3(feat3)]

class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial Attention)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r_chan = max(4, channel // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, r_chan, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r_chan, channel, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att

class PositionalEncoding(nn.Module):
    """Dynamic 2D Sinusoidal Positional Encoding"""
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        B, C, H, W = x.shape
        pos_x = torch.arange(W, dtype=torch.float32, device=x.device).repeat(H, 1)
        pos_y = torch.arange(H, dtype=torch.float32, device=x.device).repeat(W, 1).t()
        pos = torch.stack([pos_x, pos_y], dim=0)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, self.d_model, H, W, device=x.device)
        pe[0, ::2, :, :] = torch.sin(pos[0:1, :, :] * div_term[None, :, None, None])
        pe[0, 1::2, :, :] = torch.cos(pos[1:2, :, :] * div_term[None, :, None, None])
        
        return x + pe

class LightCrossAttention(nn.Module):
    """Cross-Modal Bidirectional Multi-Head Attention"""
    def __init__(self, d_model=32, num_heads=4, block_size=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        
        self.proj_q = nn.Conv2d(d_model, d_model, 1)
        self.proj_k = nn.Conv2d(d_model, d_model, 1)
        self.proj_v = nn.Conv2d(d_model, d_model, 1)
        self.proj_out = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        q_original = q
        
        pad_h = (self.block_size - (H % self.block_size)) % self.block_size
        pad_w = (self.block_size - (W % self.block_size)) % self.block_size
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))
            _, _, H_p, W_p = q.shape
            num_blocks_h = H_p // self.block_size
            num_blocks_w = W_p // self.block_size
        else:
            num_blocks_h = H // self.block_size
            num_blocks_w = W // self.block_size
            
        num_blocks = num_blocks_h * num_blocks_w

        def blockify(x):
            return x.reshape(
                B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size
            ).permute(0, 2, 4, 1, 3, 5).reshape(B * num_blocks, C, self.block_size, self.block_size)

        q_b = blockify(q)
        k_b = blockify(k)
        v_b = blockify(v)

        B_b, C_b, h, w = q_b.shape
        n = h * w
        q_proj = self.proj_q(q_b).reshape(B_b, self.num_heads, self.head_dim, n)
        k_proj = self.proj_k(k_b).reshape(B_b, self.num_heads, self.head_dim, n)
        v_proj = self.proj_v(v_b).reshape(B_b, self.num_heads, self.head_dim, n)

        attn = torch.einsum('bhdn, bhdm -> bhnm', q_proj, k_proj) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v_proj)
        out = out.contiguous().reshape(B_b, C_b, h, w)
        out = self.proj_out(out)

        out = out.reshape(
            B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(B, C, num_blocks_h * self.block_size, num_blocks_w * self.block_size)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return self.norm(out + q_original)

class LightTransformerBlock(nn.Module):
    """Global Context Transformer Block"""
    def __init__(self, d_model=32, num_heads=4, hidden_dim=64, block_size=8, dropout=0.1):
        super().__init__()
        self.self_attn = LightCrossAttention(d_model, num_heads, block_size, dropout)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, d_model, 1)
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        return x + self.dropout(self.ffn(self.norm(x)))

class RDAHNetCore(nn.Module):
    """
    Core RDAH-Net Architecture (MobileViT Dual Branch + Bidirectional Cross-Attention + PixelShuffle Decoder)
    """
    def __init__(self, d_model=32, num_heads=4, output_parameterization="unconstrained"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if output_parameterization not in {"unconstrained", "softplus"}:
            raise ValueError(
                "output_parameterization must be 'unconstrained' or 'softplus'"
            )
        self.output_parameterization = output_parameterization

        # 1. Dual MobileViT Encoders
        self.depth_encoder = MobileViT_S_Light(in_channels=1, d_model=d_model)
        self.img_encoder = MobileViT_S_Light(in_channels=3, d_model=d_model)

        # 2. CBAM Modules
        self.cbam_blocks = nn.ModuleList([CBAM(d_model) for _ in range(3)])

        # 3. Cross-Modal Bidirectional Attention
        self.cross_attn_blocks = nn.ModuleList([LightCrossAttention(d_model, num_heads, block_size=8) for _ in range(3)])
        self.rev_cross_attn_blocks = nn.ModuleList([LightCrossAttention(d_model, num_heads, block_size=8) for _ in range(3)])

        # 4. Global Transformer
        self.pos_encoding = PositionalEncoding(d_model)
        self.global_transformer = LightTransformerBlock(d_model, num_heads, hidden_dim=64)

        # 5. Skip Projections
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(d_model, 16, 1),
            nn.Conv2d(d_model, 32, 1),
            nn.Conv2d(d_model, 64, 1)
        ])

        # 6. Multi-Scale PixelShuffle Decoder (4x upsampling stages = 16x total from stage3)
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(d_model, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(16, 8 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, depth, img):
        """
        Args:
            depth: (B, 1, H, W) Relative depth prior map
            img: (B, 3, H, W) Optical RGB image
        Returns:
            height_pred: (B, 1, H, W) Continuous metric AGL height map in metres
        """
        B, _, H_in, W_in = depth.shape
        
        # 1. Multi-scale feature extraction
        depth_feats = self.depth_encoder(depth)
        img_feats = self.img_encoder(img)
        
        depth_feats = [self.cbam_blocks[i](f) for i, f in enumerate(depth_feats)]
        img_feats = [self.cbam_blocks[i](f) for i, f in enumerate(img_feats)]

        # 2. Hierarchical bidirectional cross-modal fusion
        fused_feats = []
        for i in range(3):
            feat1 = self.cross_attn_blocks[i](q=depth_feats[i], k=img_feats[i], v=img_feats[i])
            feat2 = self.rev_cross_attn_blocks[i](q=img_feats[i], k=depth_feats[i], v=depth_feats[i])
            fused_feats.append((feat1 + feat2) * 0.5)

        # 3. Global contextual transformer
        global_feat = self.pos_encoding(fused_feats[2])
        global_feat = self.global_transformer(global_feat)

        # 4. Hierarchical PixelShuffle decoding with skip connections
        x = self.decoder_stage1(global_feat)
        skip2 = F.interpolate(self.skip_projs[2](fused_feats[2]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip2

        x = self.decoder_stage2(x)
        skip1 = F.interpolate(self.skip_projs[1](fused_feats[1]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip1

        x = self.decoder_stage3(x)
        skip0 = F.interpolate(self.skip_projs[0](fused_feats[0]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip0

        height_pred = self.decoder_stage4(x)
        if self.output_parameterization == "softplus":
            height_pred = F.softplus(height_pred)
        
        # Ensure exact input dimensions
        if height_pred.shape[2:] != (H_in, W_in):
            height_pred = F.interpolate(height_pred, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return height_pred

class RDAHNetDepthWizard(nn.Module):
    """
    Unified DepthWizard Model:
    Combines frozen/fine-tuned Depth Anything V2 Backbone + RDAH-Net Metric Height Regression Head.
    """
    def __init__(self, dav2_wrapper=None, d_model=32, freeze_dav2=True):
        super().__init__()
        self.dav2_wrapper = dav2_wrapper
        self.freeze_dav2 = freeze_dav2
        self.rdah_core = RDAHNetCore(d_model=d_model)

        if self.dav2_wrapper is not None and self.freeze_dav2:
            for p in self.dav2_wrapper.model.parameters():
                p.requires_grad = False

    def forward(self, rgb, rel_depth=None):
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1]
            rel_depth: (B, 1, H, W) optional precomputed relative depth
        """
        if rel_depth is None and self.dav2_wrapper is not None:
            with torch.set_grad_enabled(not self.freeze_dav2):
                # Run DAV2 forward
                inputs = self.dav2_wrapper.processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(rgb.device) for k, v in inputs.items()}
                out = self.dav2_wrapper.model(**inputs)
                rel_depth = F.interpolate(out.predicted_depth.unsqueeze(1), size=rgb.shape[2:], mode='bilinear', align_corners=False)

        metric_height = self.rdah_core(rel_depth, rgb)
        return metric_height
