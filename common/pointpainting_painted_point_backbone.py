# -*- coding: utf-8 -*-
"""
pointpainting_painted_point_backbone

PointPainting 私有点云 Backbone 模块，复用 common.SharedMLP1D。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.shared_mlp_1d import SharedMLP1D


class PaintedPointBackbone(nn.Module):
    """
    对 PointPainting 点级编码特征执行逐点 Backbone 特征变换。

    输入:
        point_features:
            点级编码特征，shape 为 (N, 128)。

    输出:
        backbone_features:
            Backbone 高层点级特征，shape 为 (N, 256)。
    """

    def __init__(self):
        super().__init__()
        self.backbone = SharedMLP1D(
            in_channels=128,
            channels=[256, 256, 256],
            use_bn=False,
            activation=True,
            activate_last=False,
            input_layout="BLC",
            output_layout="BLC",
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        if point_features.dim() != 2:
            raise ValueError("point_features must have shape (N, C)")
        return self.backbone(point_features.unsqueeze(0)).squeeze(0)
