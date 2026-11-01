# -*- coding: utf-8 -*-
"""
IS-Fusion scene-level Camera/LiDAR BEV concat-conv 融合块。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock
from fusion_unit.symmetric.feature.ops import feature_concatenation  # noqa: F401


class ISFusionSceneBEVConcatConv(nn.Module):
    """
    将 image BEV 与 LiDAR BEV 在通道维拼接，并通过 3x3 Conv + BN + ReLU 融合。

    Args:
        image_channels: image BEV 输入通道数。
        lidar_channels: LiDAR BEV 输入通道数。
        out_channels: 输出融合 BEV 通道数。

    Forward Args:
        image_bev_feats: 图像 BEV 特征，shape 为 (B, C_img, H, W)。
        lidar_bev_feats: LiDAR BEV 特征，shape 为 (B, C_lidar, H, W)。

    Returns:
        Tensor: 融合 BEV 特征，shape 为 (B, C_out, H, W)。
    """

    def __init__(self, image_channels: int = 256, lidar_channels: int = 256, out_channels: int = 128):
        super().__init__()
        self.conv = Conv2dNormActivationBlock(
            int(image_channels) + int(lidar_channels),
            int(out_channels),
            kernel_size=3,
            use_bn=True,
            activation=True,
        )

    def forward(self, image_bev_feats: torch.Tensor, lidar_bev_feats: torch.Tensor) -> torch.Tensor:
        if image_bev_feats.shape[-2:] != lidar_bev_feats.shape[-2:]:
            raise ValueError("image_bev_feats and lidar_bev_feats must share H/W")
        fused = torch.ops.fusionrepo.feature_concatenation([image_bev_feats, lidar_bev_feats], 1)
        return self.conv(fused)
