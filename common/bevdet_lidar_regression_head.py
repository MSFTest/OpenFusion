# -*- coding: utf-8 -*-
"""BEVDet LiDAR-only sparse 3D 框回归模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class _Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([
                nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
            ])
            channels = hidden_channels
        layers.append(nn.Conv1d(channels, out_channels, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LidarOnlyRegressionHead(nn.Module):
    """
    仅使用 LiDAR sparse query 预测五组 3D 框回归量。

    Args:
        query_channels: LiDAR query 通道数。
        hidden_channels: 各回归分支隐藏通道数。
        num_layers: 每个回归分支的 Conv1d 层数。

    Inputs:
        lidar_query: LiDAR query，shape 为 ``(B,C,K)``。
        query_pos: BEV proposal 网格坐标，shape 为 ``(B,K,2)``。

    Returns:
        Dict[str, Tensor]: ``center (B,2,K)``、``height (B,1,K)``、
        ``dim (B,3,K)``、``rot (B,2,K)``、``vel (B,2,K)``。
    """

    def __init__(self, query_channels: int, hidden_channels: int = 128, num_layers: int = 2,
                 head_channels: int = 64):
        super().__init__()
        del hidden_channels
        self.center_head = _Head(query_channels, 2, head_channels, num_layers)
        self.height_head = _Head(query_channels, 1, head_channels, num_layers)
        self.dim_head = _Head(query_channels, 3, head_channels, num_layers)
        self.rot_head = _Head(query_channels, 2, head_channels, num_layers)
        self.vel_head = _Head(query_channels, 2, head_channels, num_layers)

    def forward(self, lidar_query: torch.Tensor, query_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "center": self.center_head(lidar_query) + query_pos.transpose(1, 2).to(lidar_query.dtype),
            "height": self.height_head(lidar_query),
            "dim": self.dim_head(lidar_query),
            "rot": self.rot_head(lidar_query),
            "vel": self.vel_head(lidar_query),
        }
