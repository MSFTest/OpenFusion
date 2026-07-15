# -*- coding: utf-8 -*-
"""Feature alignment module for EPNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from common.point_feature_fc_align import PointFeatureFCAlign


class FeatureAlignmentModule(nn.Module):
    """
    对齐点特征与图像点特征到统一通道数。

    Inputs:
        point_features: (B, C_p, N)，LiDAR 点特征或点坐标派生特征。
        image_point_features: (B, C_i, N)，投影采样得到的图像点特征。

    Returns:
        dict:
            point_aligned: (B, C_out, N)
            image_aligned: (B, C_out, N)

    调用算子:
        common.point_feature_fc_align.PointFeatureFCAlign。
    """

    def __init__(self, point_in_channels: int, image_in_channels: int, out_channels: int):
        super().__init__()
        self.point_align = PointFeatureFCAlign(point_in_channels, out_channels)
        self.image_align = PointFeatureFCAlign(image_in_channels, out_channels)

    def forward(
        self, point_features: torch.Tensor, image_point_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "point_aligned": self.point_align(point_features),
            "image_aligned": self.image_align(image_point_features),
        }
