# -*- coding: utf-8 -*-
"""
FrustumPointNetEncodingModule

功能：
    对 frustum 内点云做质心归一化、共享 MLP 编码和 masked max pooling，
    输出 frustum-level 特征序列。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.shared_mlp_1d import SharedMLP1D


class FrustumPointNetEncodingModule(nn.Module):
    """
    Frustum PointNet 编码模块。

    Args:
        in_channels: 点输入通道数。
        channels: 共享 MLP 输出通道序列。

    Inputs:
        frustum_points: (P, T, M, 3)
        frustum_mask: (P, T, M)

    Returns:
        frustum_feats: (P, T, C)
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()
        paper_channels = ((64, 64, 128), (64, 64, 128), (128, 128, 256), (256, 256, 512))
        self.shared_mlps = nn.ModuleList(
            [
                SharedMLP1D(
                    in_channels=in_channels,
                    channels=scale_channels,
                    use_bn=True,
                    activation=True,
                    activate_last=True,
                    input_layout="BTMC",
                    output_layout="BTMC",
                )
                for scale_channels in paper_channels
            ]
        )
        self.out_channels = tuple(scale[-1] for scale in paper_channels)

    def forward(
        self,
        multiscale_points: list[torch.Tensor],
        multiscale_masks: list[torch.Tensor],
        multiscale_refs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        outputs = []
        for points, mask, refs, mlp in zip(
            multiscale_points, multiscale_masks, multiscale_refs, self.shared_mlps
        ):
            relative_points = points - refs.unsqueeze(2)
            point_feats = mlp(relative_points)
            outputs.append(
                torch.ops.fusionrepo.masked_reduce_pool(point_feats, mask, 2, "max", False)
            )
        return outputs
