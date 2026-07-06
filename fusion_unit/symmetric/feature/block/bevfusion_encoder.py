# -*- coding: utf-8 -*-
"""
BEVfusion Camera/LiDAR BEV 融合编码模块。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class BEVFusionEncoder(nn.Module):
    """
    对拼接后的多模态 BEV 特征执行二维卷积融合编码。

    Args:
        in_channels: int
            输入 BEV 特征通道数。
        mid_channels: int
            中间卷积通道数。
        out_channels: int
            输出 BEV 特征通道数。
        use_residual: bool
            是否启用残差连接。

    Inputs:
        x: (B, C_in, H, W)
            Camera BEV 与 LiDAR BEV 拼接后的特征。

    Returns:
        fused_bev: (B, C_out, H, W)
            融合编码后的 BEV 特征。
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.conv1 = Conv2dNormActivationBlock(
            int(in_channels),
            int(mid_channels),
            kernel_size=3,
            use_bn=True,
            activation=True,
        )
        self.conv2 = Conv2dNormActivationBlock(
            int(mid_channels),
            int(out_channels),
            kernel_size=3,
            use_bn=True,
            activation=False,
        )
        self.res_conv = None
        if self.use_residual and int(in_channels) != int(out_channels):
            self.res_conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.conv1(x))
        if self.use_residual:
            if self.res_conv is not None:
                identity = self.res_conv(identity)
            out = out + identity
        return self.relu(out)
