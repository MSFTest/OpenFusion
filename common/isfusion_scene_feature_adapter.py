# -*- coding: utf-8 -*-
"""
IS-Fusion scene feature adapter。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class ISFusionSceneFeatureAdapter(nn.Module):
    """
    用 3x3 Conv + BN + ReLU 对 BEV scene feature 做通道与局部上下文适配。

    Args:
        in_channels: 输入 BEV 通道数。
        out_channels: 输出 BEV 通道数。

    Forward Args:
        bev_feats: BEV 特征，shape 为 (B, C_in, H, W)，坐标系为 ego BEV grid。

    Returns:
        Tensor: 适配后的 scene feature，shape 为 (B, C_out, H, W)。
    """

    def __init__(self, in_channels: int = 128, out_channels: int = 128):
        super().__init__()
        self.adapter = Conv2dNormActivationBlock(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            use_bn=True,
            activation=True,
        )

    def forward(self, bev_feats: torch.Tensor) -> torch.Tensor:
        return self.adapter(bev_feats)
