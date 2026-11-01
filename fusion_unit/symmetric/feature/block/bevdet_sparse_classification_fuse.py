# -*- coding: utf-8 -*-
"""BEVDet 三路 sparse query 分类融合模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseClassificationFuseNet(nn.Module):
    """
    融合 LiDAR、image-view 和 image-BEV sparse query 并预测类别。

    Args:
        in_channels: 三路 query 拼接后的通道数。
        hidden_channels: 融合隐藏通道数。
        num_classes: 检测类别数。
        num_layers: 前置融合 Conv1d 层数。

    Inputs:
        sparse_fuse_input: 多源 query，shape 为 ``(B,Cin,K)``。

    Returns:
        Tensor: proposal 分类 logits，shape 为 ``(B,num_classes,K)``。
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, num_layers: int = 2,
                 head_channels: int = 64):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_channels, num_classes, kernel_size=1),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, sparse_fuse_input: torch.Tensor) -> torch.Tensor:
        return self.layers(sparse_fuse_input)
