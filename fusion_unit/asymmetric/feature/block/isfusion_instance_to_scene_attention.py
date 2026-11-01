# -*- coding: utf-8 -*-
"""
IS-Fusion IGF instance-to-scene attention 融合块。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ISFusionInstanceToSceneAttention(nn.Module):
    """
    让 BEV grid token 从 instance token 中获取上下文信息，并叠加 scene gate residual。

    Args:
        embed_dims: BEV token 与 instance token 通道数。
        num_heads: 多头注意力 head 数。
        dropout: dropout 比例。

    Forward Args:
        bev_tokens: BEV tokens，shape 为 (B, C, H*W)。
        instance_tokens: instance tokens，shape 为 (B, C, K)。
        scene_feats: scene BEV feature，shape 为 (B, C, H, W)。

    Returns:
        Tensor: instance-aware scene tokens，shape 为 (B, C, H*W)。
    """

    def __init__(self, embed_dims: int = 128, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(int(embed_dims), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(int(embed_dims))
        self.scene_gate = nn.Sequential(
            nn.Conv2d(int(embed_dims), int(embed_dims), kernel_size=1),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(int(embed_dims), int(embed_dims) * 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dims) * 2, int(embed_dims)),
        )
        self.norm2 = nn.LayerNorm(int(embed_dims))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, bev_tokens: torch.Tensor, instance_tokens: torch.Tensor, scene_feats: torch.Tensor) -> torch.Tensor:
        query = bev_tokens.transpose(1, 2).contiguous()
        key_value = instance_tokens.transpose(1, 2).contiguous()
        attended, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        x = self.norm1(query + self.dropout(attended))
        gate = self.scene_gate(scene_feats).flatten(2).transpose(1, 2)
        scene_residual = scene_feats.flatten(2).transpose(1, 2)
        x = x + gate * scene_residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x.transpose(1, 2).contiguous()
