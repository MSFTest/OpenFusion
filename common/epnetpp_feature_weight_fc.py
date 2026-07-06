# -*- coding: utf-8 -*-
"""
EPNet++ 点级融合特征权重 logits 生成层。

该层保留 EPNet++ 原始 ILFusionWeightFC 语义：对点级融合特征执行两层
1x1 Conv1d MLP，输出未归一化权重 logits。权重激活和应用由后续门控算子完成。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .shared_mlp_1d import SharedMLP1D


class ILFusionWeightFC(nn.Module):
    """
    EPNet++ 点级权重 logits 生成层。

    Args:
        in_channels: int
            输入融合特征通道数。
        hidden_channels: int
            中间隐藏通道数。
        out_channels: int
            输出权重 logits 通道数，通常为 1。

    Shape:
        输入 fusion_features 为 (B, C_in, N)，输出为 (B, C_out, N)。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
    ):
        super().__init__()
        self.weight_fc = SharedMLP1D(
            in_channels=int(in_channels),
            channels=[int(hidden_channels), int(out_channels)],
            use_bn=False,
            activation=True,
            activate_last=False,
            input_layout="BCL",
        )

    def forward(self, fusion_features: torch.Tensor) -> torch.Tensor:
        return self.weight_fc(fusion_features)
