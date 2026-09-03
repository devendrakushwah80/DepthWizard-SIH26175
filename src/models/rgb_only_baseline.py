"""
DepthWizard (SIH26175) — RGB-Only Metric Height Baseline (Baseline B1)
Player 1: AI/ML Lead

Comparable baseline model with identical MobileViT-S encoder + PixelShuffle decoder,
operating exclusively on optical RGB images without the Depth Anything V2 geometric prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.rdah_net import MobileViT_S_Light, CBAM, PositionalEncoding, LightTransformerBlock

class RGBOnlyHeightNet(nn.Module):
    """
    RGB-Only Single-View Height Estimation Network (Baseline B1).
    Used as an experimental control to benchmark the exact marginal utility of the DAV2 depth prior.
    """
    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 1. Single RGB MobileViT-S Encoder
        self.img_encoder = MobileViT_S_Light(in_channels=3, d_model=d_model)

        # 2. CBAM Modules on RGB features
        self.cbam_blocks = nn.ModuleList([CBAM(d_model) for _ in range(3)])

        # 3. Positional Encoding + Transformer Bottleneck
        self.pos_encoding = PositionalEncoding(d_model)
        self.global_transformer = LightTransformerBlock(d_model, num_heads, hidden_dim=64)

        # 4. Skip Projections
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(d_model, 16, 1),
            nn.Conv2d(d_model, 32, 1),
            nn.Conv2d(d_model, 64, 1)
        ])

        # 5. PixelShuffle Decoder
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

    def forward(self, img):
        B, _, H_in, W_in = img.shape
        
        # Extract RGB features
        img_feats = self.img_encoder(img)
        img_feats = [self.cbam_blocks[i](f) for i, f in enumerate(img_feats)]

        # Global contextual transformer
        global_feat = self.pos_encoding(img_feats[2])
        global_feat = self.global_transformer(global_feat)

        # Decode with skip connections
        x = self.decoder_stage1(global_feat)
        skip2 = F.interpolate(self.skip_projs[2](img_feats[2]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip2

        x = self.decoder_stage2(x)
        skip1 = F.interpolate(self.skip_projs[1](img_feats[1]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip1

        x = self.decoder_stage3(x)
        skip0 = F.interpolate(self.skip_projs[0](img_feats[0]), size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip0

        height_pred = self.decoder_stage4(x)
        
        if height_pred.shape[2:] != (H_in, W_in):
            height_pred = F.interpolate(height_pred, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return height_pred
