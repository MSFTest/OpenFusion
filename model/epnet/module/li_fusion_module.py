# -*- coding: utf-8 -*-
"""LI-Fusion module for EPNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import fusion_unit  # noqa: F401 - registers fusionrepo operators
from common.epnet_feature_weight_fc import FeatureWeightFC


class LIFusionModule(nn.Module):
    """
    执行 EPNet 的点云-图像逐点特征融合。

    Inputs:
        point_aligned: (B, C, N)，对齐后的 LiDAR 点特征。
        image_aligned: (B, C, N)，对齐后的图像点特征。

    Returns:
        dict:
            fused_features: (B, C, N)，融合后的点特征。
            fusion_weight: (B, C, N)，图像门控权重。
            additive_features: (B, C, N)，加法融合中间结果。

    调用算子:
        fusionrepo::epnet_li_fusion_add、
        fusionrepo::feature_gate_multiply、
        fusionrepo::epnet_li_fusion_muti、
        fusionrepo::epnet_li_fusion_concat、
        common.epnet_feature_weight_fc.FeatureWeightFC。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight_fc = FeatureWeightFC(channels)
        self.out_proj = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, point_aligned: torch.Tensor, image_aligned: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        add_feat = torch.ops.fusionrepo.epnet_li_fusion_add(
            point_aligned, image_aligned, True
        )
        weight = self.weight_fc(add_feat)
        image_gated = torch.ops.fusionrepo.feature_gate_multiply(
            image_aligned, weight, False
        )
        point_gated = torch.ops.fusionrepo.epnet_li_fusion_muti(
            point_aligned, 1.0 - weight, False
        )
        cat = torch.ops.fusionrepo.epnet_li_fusion_concat(
            [add_feat, image_gated, point_gated], 1
        )
        return {
            "fused_features": self.out_proj(cat),
            "fusion_weight": weight,
            "additive_features": add_feat,
        }
