# -*- coding: utf-8 -*-
"""BEVDet Camera/LiDAR dense BEV 融合 heatmap 模块。"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        first = Conv2dNormActivationBlock(channels, channels, 3, use_bn=True, activation=True)
        second = Conv2dNormActivationBlock(channels, channels, 3, use_bn=True, activation=False)
        self.layers = nn.Sequential(*first.block.children(), *second.block.children())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.layers(x))


class _ResidualProjectionBlock(nn.Module):
    """与 DAL CustomResNet 首个 BasicBlock 一致的通道投影残差块。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class DenseHeatmapFuseNet(nn.Module):
    """
    对已拼接的 Camera/LiDAR dense BEV 特征进行卷积融合并预测 heatmap。

    Args:
        in_channels: 拼接后输入通道数。
        hidden_channels: 融合隐藏通道数。
        num_classes: 检测类别数。
        num_blocks: 残差融合块数量。

    Inputs:
        dense_fuse_input: 对齐 BEV 特征，shape 为 ``(B,Cin,H,W)``。

    Returns:
        Tensor: dense heatmap logits，shape 为 ``(B,num_classes,H,W)``。
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, num_blocks: int = 2):
        super().__init__()
        self.fuse = nn.Sequential(
            _ResidualProjectionBlock(in_channels, hidden_channels),
            *[_ResidualBlock(hidden_channels) for _ in range(num_blocks)],
        )
        head_stem = Conv2dNormActivationBlock(
            hidden_channels, hidden_channels, 3, use_bn=True, activation=True
        )
        self.heatmap_head = nn.Sequential(
            *head_stem.block.children(),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, dense_fuse_input: torch.Tensor) -> torch.Tensor:
        return self.heatmap_head(self.fuse(dense_fuse_input))
