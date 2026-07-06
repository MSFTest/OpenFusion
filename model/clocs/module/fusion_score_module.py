# -*- coding: utf-8 -*-
"""
FusionScoreModule

功能：
    使用 FusionMLP 为每个 pair 生成融合分数

输入：
    pair_features: (P, C) 或 (P, C, 1, 1)

输出：
    pair_scores: (P,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion_unit.asymmetric.decision.block.FusionMLP import FusionMLP


class FusionScoreModule(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.mlp = FusionMLP(in_channels=in_channels)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        # P=0 直接返回空分数，后续 scatter/聚合能正常跑
        if pair_features.numel() == 0:
            return pair_features.new_zeros((0,), dtype=torch.float32)

        # 适配 FusionMLP(Conv2d) 输入
        if pair_features.dim() == 2:
            pair_features = pair_features.unsqueeze(-1).unsqueeze(-1)  # (P,C,1,1)

        out = self.mlp(pair_features)              # (P,1,1,1) or (P,2,1,1) ...
        out = out.view(out.shape[0], -1)           # (P,D)

        # 输出统一成 (P,)
        if out.shape[1] == 1:
            scores = out[:, 0]
        else:
            # 二分类：取正类概率
            scores = F.softmax(out[:, :2], dim=1)[:, 1]

        return scores.contiguous()