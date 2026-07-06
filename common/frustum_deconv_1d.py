# -*- coding: utf-8 -*-
"""
frustum_deconv_1d

说明：
    F-ConvNet frustum 序列 1D 反卷积上采样块。
    该模块保留原始 ConvTranspose1d + BatchNorm1d + ReLU 功能逻辑，
    用于将低分辨率 frustum 序列特征上采样到目标长度。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrustumDeconv1D(nn.Module):
    """
    Frustum 序列 1D 反卷积上采样块。

    Args:
        in_channels: int
            输入通道数。

        out_channels: int
            输出通道数。

        kernel_size: int
            ConvTranspose1d 卷积核大小。

        stride: int
            ConvTranspose1d 步长。

        padding: int
            ConvTranspose1d padding。

    Shape:
        输入 x: (B, C_in, L_low)
        输出 y: (B, C_out, L_high)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L_low)

        Returns:
            y: (B, C_out, L_high)
        """
        return self.relu(self.bn(self.deconv(x)))


FrustumDeconv = FrustumDeconv1D
