# -*- coding: utf-8 -*-
"""
IS-Fusion IGF instance context aggregation 融合块。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.token_self_attention_ffn_block import TokenSelfAttentionFFNBlock


class ISFusionInstanceContextAggregation(nn.Module):
    """
    实例间 self-attention 与局部 BEV 上下文采样聚合。

    Args:
        embed_dims: instance token 通道数。
        num_heads: 多头注意力 head 数。
        num_points: 每个实例周围的可学习采样点数。
        dropout: dropout 比例。

    Forward Args:
        instance_tokens: 实例 token，shape 为 (B, C, K)。
        query_pos: 实例 BEV 位置，shape 为 (B, K, 2)，坐标为 grid 单位。
        bev_pos: BEV 位置表，shape 为 (B, H*W, 2)，保留用于接口对齐。
        scene_feats: BEV scene feature，shape 为 (B, C, H, W)。

    Returns:
        Tensor: 上下文增强实例 token，shape 为 (B, C, K)。
    """

    def __init__(self, embed_dims: int = 128, num_heads: int = 8, num_points: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_points = int(num_points)
        self.self_block = TokenSelfAttentionFFNBlock(embed_dims, num_heads, ffn_ratio=2, dropout=dropout, activation="relu")
        self.offset_head = nn.Linear(int(embed_dims), self.num_points * 2)
        self.weight_head = nn.Linear(int(embed_dims), self.num_points)
        self.value_proj = nn.Conv2d(int(embed_dims), int(embed_dims), kernel_size=1)
        self.out_proj = nn.Linear(int(embed_dims), int(embed_dims))
        self.norm = nn.LayerNorm(int(embed_dims))
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        instance_tokens: torch.Tensor,
        query_pos: torch.Tensor,
        bev_pos: torch.Tensor,
        scene_feats: torch.Tensor,
    ) -> torch.Tensor:
        tokens = instance_tokens.transpose(1, 2).contiguous()
        tokens = self.self_block(tokens)
        batch_size, _, height, width = scene_feats.shape
        offsets = torch.tanh(self.offset_head(tokens)).view(batch_size, tokens.shape[1], self.num_points, 2)
        weights = F.softmax(self.weight_head(tokens), dim=-1)

        base = query_pos.to(scene_feats.dtype)
        norm_y = base[..., 0] / max(height - 1, 1) * 2.0 - 1.0
        norm_x = base[..., 1] / max(width - 1, 1) * 2.0 - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)
        grid = grid + offsets * (2.0 / max(height, width))
        grid = grid.reshape(batch_size, tokens.shape[1] * self.num_points, 1, 2)

        context_map = self.value_proj(scene_feats)
        sampled = F.grid_sample(context_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(-1).view(batch_size, scene_feats.shape[1], tokens.shape[1], self.num_points)
        context = (sampled * weights[:, None]).sum(dim=-1).transpose(1, 2)
        tokens = self.norm(tokens + self.dropout(self.out_proj(context)))
        return tokens.transpose(1, 2).contiguous()
