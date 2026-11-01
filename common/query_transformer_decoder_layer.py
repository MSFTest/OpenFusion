# -*- coding: utf-8 -*-
"""Sparse query 使用的公共 Transformer decoder layer。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class QueryTransformerDecoderLayer(nn.Module):
    """执行可选 self-attention、cross-attention 与 FFN。

    输入 query/key 为 ``[B,C,Nq/Nk]``，位置为位置编码器声明的
    ``[B,N,D]``，输出 ``[B,C,Nq]``。运算顺序、position 同时加入
    Q/K/V、三组 residual/dropout/post-LayerNorm 与 SparseFusion
    ``transformerdecoder.py`` 和 DeepInteraction ``decoder_utils.py`` 一致。
    ``cross_only=True`` 时跳过 self-attention，但仍保留原 norm 参数以兼容
    SparseFusion checkpoint。
    """

    def __init__(self, hidden_channels: int, num_heads: int,
                 feedforward_channels: int = 256, dropout: float = 0.1,
                 activation: str = "relu", query_position: Optional[nn.Module] = None,
                 key_position: Optional[nn.Module] = None, cross_only: bool = False):
        super().__init__()
        self.cross_only = bool(cross_only)
        if not self.cross_only:
            self.self_attn = nn.MultiheadAttention(hidden_channels, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_channels, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_channels, feedforward_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if activation not in ("relu", "gelu"):
            raise ValueError("activation 必须是 relu 或 gelu")
        self.activation = F.relu if activation == "relu" else F.gelu
        self.self_posembed = query_position
        self.cross_posembed = key_position

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                query_pos: torch.Tensor, key_pos: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        qpos = None if self.self_posembed is None else self.self_posembed(query_pos).permute(2, 0, 1)
        kpos = None if self.cross_posembed is None else self.cross_posembed(key_pos).permute(2, 0, 1)
        query, key = query.permute(2, 0, 1), key.permute(2, 0, 1)
        add = lambda tensor, position: tensor if position is None else tensor + position
        if not self.cross_only:
            positioned = add(query, qpos)
            update = self.self_attn(positioned, positioned, positioned)[0]
            query = self.norm1(query + self.dropout1(update))
        update, weights = self.multihead_attn(
            add(query, qpos), add(key, kpos), add(key, kpos), attn_mask=attn_mask
        )
        query = self.norm2(query + self.dropout2(update))
        update = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = self.norm3(query + self.dropout3(update)).permute(1, 2, 0)
        return (query, weights) if need_weights else query
