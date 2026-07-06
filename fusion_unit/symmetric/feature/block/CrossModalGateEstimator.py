# -*- coding: utf-8 -*-
"""
CrossModalGateEstimator

跨模态门控 / 注意力权重生成模块。

该模块接收已经空间或索引对齐的多路特征，先在通道维拼接，再通过
轻量投影网络生成 gate、attention logits 或归一化权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalGateEstimator(nn.Module):
    """
    跨模态 gate / attention 权重估计器。

    Args:
        in_channels:
            单路输入特征通道数。

        num_inputs:
            输入特征路数。

        hidden_channels:
            中间通道数。

        out_channels:
            输出权重通道数。

        layout:
            输入布局，支持 "BCHW"、"BCN"、"BNC"。

        activation:
            输出激活，支持 "sigmoid"、"softmax_modal"、"none"。

    输入:
        features:
            多路特征列表。BCHW 布局下每路 shape 为 (B, C, H, W)，
            BCN 布局下为 (B, C, N)，BNC 布局下为 (B, N, C)。

    输出:
        weights:
            gate 或 attention 权重，shape 由 layout 与 out_channels 决定。
    """

    def __init__(
        self,
        in_channels: int,
        num_inputs: int,
        hidden_channels: int,
        out_channels: int,
        layout: str = "BCHW",
        activation: str = "sigmoid",
    ):
        super().__init__()
        self.layout = str(layout).upper()
        self.activation = str(activation).lower()

        total_in = int(in_channels) * int(num_inputs)
        hidden = int(hidden_channels)
        out = int(out_channels)

        if self.layout == "BCHW":
            self.net = nn.Sequential(
                nn.Conv2d(total_in, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out, kernel_size=1, bias=True),
            )
            self.concat_dim = 1
        elif self.layout == "BCN":
            self.net = nn.Sequential(
                nn.Conv1d(total_in, hidden, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, out, kernel_size=1, bias=True),
            )
            self.concat_dim = 1
        elif self.layout == "BNC":
            self.net = nn.Sequential(
                nn.Linear(total_in, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, out, bias=True),
            )
            self.concat_dim = -1
        else:
            raise ValueError("layout must be one of 'BCHW', 'BCN', 'BNC'")

    def forward(self, features) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError("features must be a non-empty list or tuple")

        fused = torch.cat(list(features), dim=self.concat_dim)
        logits = self.net(fused)

        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.activation == "softmax_modal":
            dim = 1 if self.layout in ("BCHW", "BCN") else -1
            return torch.softmax(logits, dim=dim)
        if self.activation == "none":
            return logits

        raise ValueError("activation must be one of 'sigmoid', 'softmax_modal', 'none'")
