# -*- coding: utf-8 -*-
"""
CenterFusion 图像侧预测头。

本文件保留 CenterFusion 原始 6 个可学习图像预测头的功能与输入输出语义，
仅统一放置目录、命名风格和中文说明。所有模块均为 nn.Module，不注册为
torch.library 算子。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CenterFusionConvHead(nn.Module):
    """
    CenterFusion 图像预测头基础结构。

    Args:
        in_channels: 输入图像特征通道数。
        out_channels: 输出预测通道数。
        hidden_channels: 中间卷积通道数。

    Returns:
        Tensor:
            输出形状为 (B, out_channels, H, W)。
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)


class CenterHeatmapHead(_CenterFusionConvHead):
    """
    中心点热力图预测头。

    Args:
        feat: (B, C_in, H, W)

    Returns:
        heatmap: (B, num_classes, H, W)
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 64):
        super().__init__(in_channels, num_classes, hidden_channels)


class Box2DSizeHead(_CenterFusionConvHead):
    """
    2D 框尺寸回归头。

    Args:
        feat: (B, C_in, H, W)

    Returns:
        wh: (B, 2, H, W)，通道为 (width, height)。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__(in_channels, 2, hidden_channels)


class CenterOffsetHead(_CenterFusionConvHead):
    """
    中心点偏移回归头。

    Args:
        feat: (B, C_in, H, W)

    Returns:
        offset: (B, 2, H, W)，通道为 (offset_x, offset_y)。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__(in_channels, 2, hidden_channels)


class DepthRegressionHead(_CenterFusionConvHead):
    """
    目标深度回归头。

    Args:
        feat: (B, C_in, H, W)

    Returns:
        depth: (B, 1, H, W)。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__(in_channels, 1, hidden_channels)


class Dimension3DHead(_CenterFusionConvHead):
    """
    三维尺寸回归头。

    Args:
        feat: (B, C_in, H, W)

    Returns:
        dims: (B, 3, H, W)，通道为 (length, width, height)。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__(in_channels, 3, hidden_channels)


class RotationRegressionHead(_CenterFusionConvHead):
    """
    朝向角回归头。

    Args:
        feat: (B, C_in, H, W)

    Returns:
        rot: (B, out_channels, H, W)。
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64):
        super().__init__(in_channels, out_channels, hidden_channels)
