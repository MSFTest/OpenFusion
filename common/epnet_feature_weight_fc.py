# -*- coding: utf-8 -*-
"""
EPNeT 点级特征权重生成层。

该层保留 EPNeT 原始 FeatureWeightFC 语义：对点级融合特征执行 1x1
通道映射，并通过 sigmoid 生成 [0, 1] 权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .shared_mlp_1d import SharedMLP1D


class FeatureWeightFC(nn.Module):
    """
    EPNeT 特征权重生成层。

    Args:
        in_dim: int
            输入融合特征通道数。
        out_dim: int
            输出权重通道数，默认 1。
        use_bn: bool
            是否在 1x1 映射后使用 BatchNorm1d。

    Shape:
        输入 H 为 (B, C, N)，输出权重为 (B, out_dim, N)。
    """

    def __init__(self, in_dim: int, out_dim: int = 1, use_bn: bool = False):
        super().__init__()
        self.fc = SharedMLP1D(
            in_channels=int(in_dim),
            channels=[int(out_dim)],
            use_bn=bool(use_bn),
            activation=False,
            activate_last=bool(use_bn),
            input_layout="BCL",
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(H))
