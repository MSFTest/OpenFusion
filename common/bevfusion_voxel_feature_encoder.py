# -*- coding: utf-8 -*-
"""
BEVfusion 体素特征编码模块。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.VoxelPointwiseEncoder import VoxelPointwiseEncoder


class BEVFusionVoxelFeatureEncoder(nn.Module):
    """
    体素内逐点编码与池化模块。

    Args:
        in_channels: int
            输入点特征通道数。
        mid_channels: int
            中间点级特征通道数。
        out_channels: int
            输出体素特征通道数。
        use_second_linear: bool
            是否启用第二层共享 MLP。

    Inputs:
        voxels: (M, T, C_in)
            体素内点特征。
        num_points_per_voxel: (M,)
            每个体素真实点数。

    Returns:
        voxel_feats: (M, C_out)
            编码后的体素特征。
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_second_linear: bool = True,
    ):
        super().__init__()
        channels = [int(mid_channels), int(out_channels)] if use_second_linear else [int(mid_channels)]
        self.encoder = VoxelPointwiseEncoder(
            in_channels=int(in_channels),
            channels=channels,
            use_bn=True,
            pool_mode="max",
        )

    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(voxels, num_points_per_voxel=num_points_per_voxel)


VoxelFeatureEncoder = BEVFusionVoxelFeatureEncoder
