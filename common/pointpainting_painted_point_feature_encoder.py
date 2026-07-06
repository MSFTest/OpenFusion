# -*- coding: utf-8 -*-
"""
pointpainting_painted_point_feature_encoder

PointPainting 私有特征编码模块，复用 common.SharedMLP1D。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.shared_mlp_1d import SharedMLP1D


class PaintedPointFeatureEncoder(nn.Module):
    """
    对完成语义拼接后的点云特征执行逐点 MLP 编码。

    Args:
        in_channels:
            输入 painted point 特征维度 D + C。

    输入:
        painted_points:
            融合后的点云特征，shape 为 (N, in_channels)。

    输出:
        encoded_points:
            编码后的点级特征，shape 为 (N, 128)。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.mlp = SharedMLP1D(
            in_channels=int(in_channels),
            channels=[64, 128, 128],
            use_bn=False,
            activation=True,
            activate_last=False,
            input_layout="BLC",
            output_layout="BLC",
        )

    def forward(self, painted_points: torch.Tensor) -> torch.Tensor:
        if painted_points.dim() != 2:
            raise ValueError("painted_points must have shape (N, C)")
        return self.mlp(painted_points.unsqueeze(0)).squeeze(0)
