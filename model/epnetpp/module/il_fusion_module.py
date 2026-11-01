# -*- coding: utf-8 -*-
"""EPNet++ point-to-image IL-Fusion module."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import fusion_unit  # noqa: F401 - registers fusionrepo operators
import utils  # noqa: F401 - registers fusionrepo operators
from common.epnetpp_conv2d_fusion import ILFusionConv2DFusion
from common.epnetpp_feature_weight_fc import ILFusionWeightFC


class ILFusionModule(nn.Module):
    """
    将 LiDAR 点特征散射到图像/网格空间并增强图像特征。

    Args:
        point_aligned: Tensor，shape 为 (B, C, N)，LiDAR 点特征。
        image_point_aligned: Tensor，shape 为 (B, C, N)，点位图像特征。
        image_grid_features: Tensor，shape 为 (B, C, H, W)，图像网格特征。
        point_grid_idx: Tensor，shape 为 (B, N, 2)，图像/网格整数坐标，顺序为 (y, x)。

    Returns:
        dict:
            enhanced_image_features: (B, C, H, W)
            p2i_weight: (B, 1, N)
            scattered_point_features: (B, C, H, W)
    """

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, int(channels) // 2)
        self.weight_fc = ILFusionWeightFC(channels, hidden, 1)
        self.conv_fusion = ILFusionConv2DFusion(channels * 2, channels, kernel_size=3, use_bn=True)

    def forward(
        self,
        point_aligned: torch.Tensor,
        image_point_aligned: torch.Tensor,
        image_grid_features: torch.Tensor,
        point_grid_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        add_feat = torch.ops.fusionrepo.epnet_li_fusion_add(point_aligned, image_point_aligned, True)
        weight_logits = self.weight_fc(add_feat)
        point_gated = torch.ops.fusionrepo.epnet_li_fusion_muti(point_aligned, weight_logits, True)
        scatter_input = point_gated.transpose(1, 2).contiguous()
        scattered = torch.ops.fusionrepo.epnetpp_point_to_grid_scatter(
            scatter_input,
            point_grid_idx,
            int(image_grid_features.shape[-2]),
            int(image_grid_features.shape[-1]),
        )
        concat = torch.ops.fusionrepo.epnet_li_fusion_concat([image_grid_features, scattered], 1)
        return {
            "enhanced_image_features": self.conv_fusion(concat),
            "p2i_weight": torch.sigmoid(weight_logits),
            "scattered_point_features": scattered,
        }
