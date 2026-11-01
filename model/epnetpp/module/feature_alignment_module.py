# -*- coding: utf-8 -*-
"""EPNet++ point/image point feature alignment module."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from common.point_feature_fc_align import PointFeatureFCAlign


class FeatureAlignmentModule(nn.Module):
    """
    对齐 LiDAR 点特征与点位图像特征到统一融合维度。

    Args:
        point_features: Tensor，shape 为 (B, C_p, N)。
        image_point_features: Tensor，shape 为 (B, C_i, N)。

    Returns:
        dict:
            point_aligned: (B, C_out, N)
            image_aligned: (B, C_out, N)
    """

    def __init__(self, point_in_channels: int, image_in_channels: int, out_channels: int):
        super().__init__()
        self.point_align = PointFeatureFCAlign(point_in_channels, out_channels)
        self.image_align = PointFeatureFCAlign(image_in_channels, out_channels)

    def forward(
        self,
        point_features: torch.Tensor,
        image_point_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "point_aligned": self.point_align(point_features),
            "image_aligned": self.image_align(image_point_features),
        }
