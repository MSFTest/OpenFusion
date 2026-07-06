# -*- coding: utf-8 -*-
"""
VoxelPointwiseEncoder

通用体素内点级特征编码模块。

该模块用于对每个体素内部的点特征执行共享 MLP 映射，并在体素内部
做 max / mean / sum 池化，输出每个体素一个定长特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.shared_mlp_1d import SharedMLP1D


class VoxelPointwiseEncoder(nn.Module):
    """
    体素内逐点编码 + 分组池化模块。

    Args:
        in_channels:
            输入点特征通道数。

        channels:
            共享 MLP 每层输出通道，最后一个值为输出体素特征维度。

        use_bn:
            是否在共享 MLP 中使用 BatchNorm1d。

        pool_mode:
            体素内池化方式，支持 "max"、"mean"、"sum"。

    输入:
        voxel_features:
            shape 为 (N, P, C_in)，N 为体素数量，P 为每个体素内点数。

        mask:
            可选有效点 mask，shape 为 (N, P)，True 表示有效点。

        num_points_per_voxel:
            可选每个体素真实点数，shape 为 (N,)。

    输出:
        voxel_encoded:
            shape 为 (N, C_out)。
    """

    def __init__(
        self,
        in_channels: int,
        channels,
        use_bn: bool = True,
        pool_mode: str = "max",
    ):
        super().__init__()
        self.pool_mode = str(pool_mode)
        self.point_mlp = SharedMLP1D(
            in_channels=int(in_channels),
            channels=list(channels),
            use_bn=bool(use_bn),
            activation=True,
            activate_last=True,
            input_layout="BLC",
            output_layout="BLC",
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        num_points_per_voxel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if voxel_features.dim() != 3:
            raise ValueError("voxel_features must have shape (N, P, C)")

        n, p, _ = voxel_features.shape
        encoded = self.point_mlp(voxel_features)

        if mask is None and num_points_per_voxel is not None:
            arange = torch.arange(p, device=voxel_features.device)
            mask = arange.unsqueeze(0) < num_points_per_voxel.to(device=voxel_features.device).unsqueeze(1)

        if mask is None:
            mask = torch.ones((n, p), device=voxel_features.device, dtype=torch.bool)

        return torch.ops.fusionrepo.masked_reduce_pool(
            encoded,
            mask,
            1,
            self.pool_mode,
            False,
        )
