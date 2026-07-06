# -*- coding: utf-8 -*-
"""
EPNet++ 网格级 2D 卷积融合层。

该文件保留 EPNet++ 原始 ILFusionConv2DFusion 类名和输入输出语义，内部使用
通用 Conv2dNormActivationBlock 完成二维融合特征的空间建模。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv2d_norm_activation import Conv2dNormActivationBlock


class ILFusionConv2DFusion(nn.Module):
    """
    EPNet++ 网格级二维卷积融合模块。

    Args:
        in_channels: int
            输入网格融合特征通道数。
        out_channels: int
            输出网格融合特征通道数。
        kernel_size: int
            Conv2d 卷积核大小。
        use_bn: bool
            是否使用 BatchNorm2d。

    Shape:
        输入 grid_features 为 (B, C_in, H, W)，输出为 (B, C_out, H, W)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
    ):
        super().__init__()
        self.conv_fusion = Conv2dNormActivationBlock(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            use_bn=bool(use_bn),
            activation=True,
            inplace=True,
        )

    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        return self.conv_fusion(grid_features)
