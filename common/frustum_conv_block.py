# -*- coding: utf-8 -*-
"""
frustum_conv_block

说明：
    F-ConvNet frustum 序列 Conv1d 基础块。
    该模块保留原始两层 Conv1d + BatchNorm1d + ReLU 的功能逻辑，
    用于沿 frustum 深度序列建模局部上下文。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrustumConvBlock(nn.Module):
    """
    Frustum 序列 1D 卷积块。

    Args:
        in_channels: int
            输入通道数。

        out_channels: int
            输出通道数。

        mid_channels: int | None
            中间通道数。为 None 时使用 out_channels。

        downsample: bool
            第二层卷积是否使用 stride=2 进行序列下采样。

    Shape:
        输入 x: (B, C_in, L)
        输出 y: (B, C_out, L')，downsample=True 时 L' 约为 L/2。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        downsample: bool = False,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        stride2 = 2 if downsample else 1
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=stride2, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)

        Returns:
            y: (B, C_out, L')
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
