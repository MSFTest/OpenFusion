# -*- coding: utf-8 -*-
"""EPNet++ image-to-point LI-Fusion module."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import fusion_unit  # noqa: F401 - registers fusionrepo operators
from common.epnetpp_feature_weight_fc import ILFusionWeightFC


class LIFusionModule(nn.Module):
    """
    将点位图像语义特征融合到 LiDAR 点特征。

    Args:
        point_aligned: Tensor，shape 为 (B, C, N)。
        image_aligned: Tensor，shape 为 (B, C, N)。

    Returns:
        dict:
            enhanced_point_features: (B, C, N)
            i2p_weight: (B, 1, N)
            li_additive_features: (B, C, N)
    """

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, int(channels) // 2)
        self.weight_fc = ILFusionWeightFC(channels, hidden, 1)
        self.out_proj = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, point_aligned: torch.Tensor, image_aligned: torch.Tensor) -> Dict[str, torch.Tensor]:
        add_feat = torch.ops.fusionrepo.epnet_li_fusion_add(point_aligned, image_aligned, True)
        weight_logits = self.weight_fc(add_feat)
        image_gated = torch.ops.fusionrepo.epnet_li_fusion_muti(image_aligned, weight_logits, True)
        concat = torch.ops.fusionrepo.epnet_li_fusion_concat([point_aligned, image_gated], 1)
        return {
            "enhanced_point_features": self.out_proj(concat),
            "i2p_weight": torch.sigmoid(weight_logits),
            "li_additive_features": add_feat,
        }
