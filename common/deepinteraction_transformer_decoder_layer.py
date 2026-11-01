# -*- coding: utf-8 -*-
"""DeepInteraction 初始 query Transformer decoder layer。"""

import torch.nn.functional as F
from torch import nn

from common.deepinteraction_position_embedding import PositionEmbeddingLearned


class TransformerDecoderLayer(nn.Module):
    """依次执行 self-attention、cross-attention 与 ReLU FFN。

    Args:
        channels: token 通道，正式配置 128。
        heads: 注意力头数，正式配置 8。
        feedforward: FFN 隐藏维，正式配置 256。
        dropout: 各原始 dropout 概率。
    Returns:
        query ``[B,C,K]``；key ``[B,C,HW]``，位置分别为 ``[B,K,2]`` 和
        ``[B,HW,2]``。计算顺序、残差和 LayerNorm 与原实现一致。
    原源码：``decoder_utils.py:35-113``。
    """
    def __init__(self, channels=128, heads=8, feedforward=256, dropout=.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(channels, heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(channels, heads, dropout=dropout)
        self.linear1 = nn.Linear(channels, feedforward)
        self.linear2 = nn.Linear(feedforward, channels)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.query_position = PositionEmbeddingLearned(channels)
        self.key_position = PositionEmbeddingLearned(channels)

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 decoder_utils Transformer 与位置编码 checkpoint 键。"""
        mapped = {}
        for key, value in state_dict.items():
            key = key.replace("self_attn.", "self_attention.", 1)
            key = key.replace("multihead_attn.", "cross_attention.", 1)
            key = key.replace(
                "self_posembed.position_embedding_head.", "query_position.layers.", 1
            )
            key = key.replace(
                "cross_posembed.position_embedding_head.", "key_position.layers.", 1
            )
            mapped[key] = value
        return mapped

    def forward(self, query, key, query_position, key_position):
        query_pos = self.query_position(query_position).permute(2, 0, 1)
        key_pos = self.key_position(key_position).permute(2, 0, 1)
        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)
        positioned = query + query_pos
        update = self.self_attention(positioned, positioned, positioned)[0]
        query = self.norm1(query + self.dropout1(update))
        update = self.cross_attention(query + query_pos, key + key_pos,
                                      key + key_pos)[0]
        query = self.norm2(query + self.dropout2(update))
        update = self.linear2(self.dropout(F.relu(self.linear1(query))))
        query = self.norm3(query + self.dropout3(update))
        return query.permute(1, 2, 0)
