# -*- coding: utf-8 -*-
"""可配置的一维 learned position embedding 公共层。"""

from __future__ import annotations

import torch
from torch import nn


class LearnedPositionEmbedding1d(nn.Module):
    """把位置 ``[B,N,D]`` 编码为 ``[B,C,N]``。

    结构严格为 ``Conv1d(D,C,1)-optional BN1d-ReLU-Conv1d(C,C,1)``，覆盖
    SparseFusion 的 2/3/9 维位置编码和 DeepInteraction 的 2 维位置编码。
    第一层卷积严格保留原 ``nn.Conv1d`` 默认 ``bias=True``，不能套用常见的
    Conv-BN 去 bias 优化；BN 保留 running statistics 和 train/eval 行为。
    """

    def __init__(self, input_channels: int, hidden_channels: int,
                 use_batch_norm: bool = True):
        super().__init__()
        layers = [nn.Conv1d(input_channels, hidden_channels, 1, bias=True)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.extend((nn.ReLU(inplace=True), nn.Conv1d(hidden_channels, hidden_channels, 1)))
        self.position_embedding_head = nn.Sequential(*layers)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        if position.ndim != 3 or position.shape[-1] != self.position_embedding_head[0].in_channels:
            raise ValueError("position 必须是 [B,N,input_channels]")
        return self.position_embedding_head(position.transpose(1, 2).contiguous())

    @staticmethod
    def remap_deepinteraction_state_dict(state_dict: dict) -> dict:
        """把 DeepInteraction ``layers.*`` 键映射为当前公共层键。"""
        return {key.replace("layers.", "position_embedding_head.", 1): value
                for key, value in state_dict.items()}
