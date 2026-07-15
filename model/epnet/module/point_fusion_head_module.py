# -*- coding: utf-8 -*-
"""Point fusion head for EPNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PointFusionHeadModule(nn.Module):
    """
    基于融合点特征预测可训练的点级监督信号。

    Inputs:
        fused_features: (B, C, N)，融合点特征。

    Returns:
        dict:
            point_cls_logits: (B, N)，点级前景 logits。
            point_box_delta: (B, N, 7)，点到 3D 框的回归残差。
            point_loc_conf: (B, N)，点位置置信度。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(64, in_channels)
        self.shared = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv1d(hidden, 1, 1)
        self.box = nn.Conv1d(hidden, 7, 1)
        self.loc = nn.Conv1d(hidden, 1, 1)

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(fused_features)
        return {
            "point_cls_logits": self.cls(feat).squeeze(1),
            "point_box_delta": self.box(feat).transpose(1, 2).contiguous(),
            "point_loc_conf": torch.sigmoid(self.loc(feat).squeeze(1)),
        }
