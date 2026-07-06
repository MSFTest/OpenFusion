# -*- coding: utf-8 -*-
"""
通用 2D 卷积-归一化-激活网络块。

该模块用于 BEV、图像特征图或二维融合特征图上的局部空间建模。它只提供
通用网络构建能力，不绑定任何具体融合流程。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dNormActivationBlock(nn.Module):
    """
    2D Conv + optional BatchNorm2d + optional activation 通用模块。

    Args:
        in_channels: int
            输入特征通道数。
        out_channels: int
            输出特征通道数。
        kernel_size: int
            二维卷积核大小。
        padding: int | None
            二维卷积 padding。为 None 时默认使用 kernel_size // 2，保持空间尺寸。
        use_bn: bool
            是否在 Conv2d 后使用 BatchNorm2d。
        activation: bool
            是否在归一化后使用 ReLU 激活。
        inplace: bool
            ReLU 是否使用 inplace 模式。

    Shape:
        输入 x 为 (B, C_in, H, W)，输出为 (B, C_out, H_out, W_out)。
        当 stride=1 且 padding=kernel_size//2 时，H_out=H，W_out=W。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        use_bn: bool = True,
        activation: bool = True,
        inplace: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = int(kernel_size) // 2

        layers: list[nn.Module] = [
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                padding=int(padding),
                bias=not bool(use_bn),
            )
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(int(out_channels)))
        if activation:
            layers.append(nn.ReLU(inplace=bool(inplace)))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor
                输入二维特征图，shape 为 (B, C_in, H, W)。

        Returns:
            Tensor
                输出二维特征图，shape 为 (B, C_out, H_out, W_out)。
        """
        return self.block(x)
