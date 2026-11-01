# -*- coding: utf-8 -*-
"""
SequenceFeatureModule

功能：
    将 frustum-level 特征序列转换为 Conv1d 特征图，并融合 proposal 类别、
    proposal score 与 depth bin center 等序列上下文。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.shared_mlp_1d import SharedMLP1D


class SequenceFeatureModule(nn.Module):
    """
    序列特征构建模块。

    Args:
        in_channels: frustum feature 通道数。
        out_channels: 输出 Conv1d 序列特征通道数。
        num_classes: 2D proposal 类别数。

    Inputs:
        frustum_feats: (P, T, C)
        frustum_bins: (P, T, 2)
        proposal_scores: (P,)
        proposal_labels: (P,)

    Returns:
        sequence_features: (P, out_channels, T)
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        num_classes: int = 3,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        extra_channels = 1 + 1 + self.num_classes
        self.proj = SharedMLP1D(
            in_channels=in_channels + extra_channels,
            channels=(out_channels,),
            use_bn=True,
            activation=True,
            activate_last=True,
            input_layout="BCL",
            output_layout="BCL",
        )
        self.out_channels = int(out_channels)

    def forward(
        self,
        frustum_feats: torch.Tensor,
        frustum_bins: torch.Tensor,
        proposal_scores: torch.Tensor,
        proposal_labels: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.ops.fusionrepo.build_conv1d_feature_map(frustum_feats)
        P, _, T = x.shape
        depth_center = frustum_bins.mean(dim=-1).unsqueeze(1)
        score_feat = proposal_scores.view(P, 1, 1).expand(P, 1, T).to(dtype=x.dtype, device=x.device)
        labels = proposal_labels.clamp(min=0, max=self.num_classes - 1).long()
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=x.dtype, device=x.device)
        one_hot = one_hot.unsqueeze(-1).expand(P, self.num_classes, T)
        x = torch.cat([x, depth_center.to(dtype=x.dtype), score_feat, one_hot], dim=1)
        return self.proj(x)
