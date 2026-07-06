# -*- coding: utf-8 -*-
"""
BEVfusion 稠密三维体素主干模块。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BEVFusionSparse3DBackbone(nn.Module):
    """
    将稀疏体素特征写入 dense 3D grid 后执行 3D 卷积编码。

    Args:
        in_channels: int
            输入体素特征通道数。
        hidden_channels: int
            中间 3D 卷积通道数。
        out_channels: int
            输出 3D 特征通道数。
        grid_size: tuple[int, int, int]
            三维网格尺寸 (Z, Y, X)。

    Inputs:
        voxel_feats: (M, C_in)
            稀疏体素特征。
        voxel_coords: (M, 4)
            体素坐标 [batch_idx, z, y, x]。
        batch_size: int
            batch 大小。

    Returns:
        feat_3d: (B, C_out, Z, Y, X)
            三维体素特征体。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        grid_size: Tuple[int, int, int],
    ):
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(int(hidden_channels)),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(int(hidden_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        voxel_feats: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if voxel_feats.dim() != 2:
            raise ValueError("voxel_feats must have shape (M, C)")
        if voxel_coords.dim() != 2 or voxel_coords.shape[1] != 4:
            raise ValueError("voxel_coords must have shape (M, 4)")

        z_size, y_size, x_size = self.grid_size
        dense_grid = voxel_feats.new_zeros(
            (int(batch_size), voxel_feats.shape[1], z_size, y_size, x_size)
        )
        coords = voxel_coords.to(device=voxel_feats.device, dtype=torch.long)
        b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        valid = (
            (b >= 0) & (b < int(batch_size)) &
            (z >= 0) & (z < z_size) &
            (y >= 0) & (y < y_size) &
            (x >= 0) & (x < x_size)
        )
        if valid.any():
            dense_grid[b[valid], :, z[valid], y[valid], x[valid]] = voxel_feats[valid]
        return self.conv_block2(self.conv_block1(dense_grid))


Sparse3DBackbone = BEVFusionSparse3DBackbone
