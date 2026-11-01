# -*- coding: utf-8 -*-
"""
LiDARBEVFeatureGeneration

3D-CVF 私有 LiDAR BEV 特征生成模块。

该模块将稀疏体素特征散射到 dense BEV 平面，并保留 3D-CVF 原始的
两层 Conv2d + BatchNorm2d + ReLU 空间精炼结构。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import utils.voxel_features_to_bev_scatter  # noqa: F401


class LiDARBEVFeatureGeneration(nn.Module):
    """
    LiDAR 体素特征到 BEV 特征图生成模块。

    Args:
        in_channels:
            输入体素特征通道数。

        out_channels:
            输出 BEV 特征通道数。

    输入:
        voxel_features:
            shape 为 (N, C_in)。

        voxel_coords:
            shape 为 (N, 4)，格式为 (batch_idx, z, y, x)。

        bev_shape:
            BEV 空间尺寸 (H, W)。

        batch_size:
            输出 batch 大小 B。

    输出:
        bev_features:
            shape 为 (B, C_out, H, W)。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bev_conv = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        bev_shape: tuple,
        batch_size: int,
    ) -> torch.Tensor:
        if len(bev_shape) != 2:
            raise ValueError("bev_shape must be (H, W)")
        h, w = int(bev_shape[0]), int(bev_shape[1])
        bev_feature = torch.ops.fusionrepo.voxel_features_to_bev_scatter(
            voxel_features,
            voxel_coords,
            int(batch_size),
            h,
            w,
            "assign",
        )
        return self.bev_conv(bev_feature)
