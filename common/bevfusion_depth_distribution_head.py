# -*- coding: utf-8 -*-
"""
BEVfusion 像素深度分布预测头。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class BEVFusionDepthDistributionHead(nn.Module):
    """
    像素级离散深度概率预测模块。

    Args:
        in_channels: int
            输入相机特征通道数。
        mid_channels: int
            中间卷积通道数。
        num_depth_bins: int
            离散深度 bin 数 D。
        use_bn: bool
            是否使用 BatchNorm2d。

    Inputs:
        x: (B, C_in, H, W)
            相机二维特征图。

    Returns:
        depth_prob: (B, D, H, W)
            在深度维度归一化后的概率分布。
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_depth_bins: int,
        use_bn: bool = True,
    ):
        super().__init__()
        self.stem = Conv2dNormActivationBlock(
            int(in_channels),
            int(mid_channels),
            kernel_size=3,
            use_bn=bool(use_bn),
            activation=True,
        )
        self.depth_logits = nn.Conv2d(int(mid_channels), int(num_depth_bins), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.depth_logits(self.stem(x))
        return F.softmax(logits, dim=1)


DepthDistributionHead = BEVFusionDepthDistributionHead
