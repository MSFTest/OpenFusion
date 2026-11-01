# -*- coding: utf-8 -*-
"""
通用 token 自注意力与 FFN 网络块。

该模块用于窗口 token、RoI grid token 或其他序列特征的关系建模，不绑定
具体融合流程。它包含 MultiheadAttention、残差连接、LayerNorm 与 FFN。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenSelfAttentionFFNBlock(nn.Module):
    """
    token self-attention + FFN 通用模块。

    Args:
        embed_dims: token 通道数 C。
        num_heads: 多头注意力 head 数。
        ffn_ratio: FFN 隐藏层相对通道倍率。
        dropout: dropout 比例。
        activation: FFN 激活函数，支持 "gelu" 与 "relu"。

    Forward Args:
        tokens: token 特征，shape 为 (B, N, C)。
        key_padding_mask: padding 掩码，shape 为 (B, N)，True 表示 padding。
        attn_mask: 注意力掩码，shape 由 nn.MultiheadAttention 支持。

    Returns:
        Tensor: 增强后的 token 特征，shape 为 (B, N, C)。
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(int(embed_dims), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(int(embed_dims))
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError("activation must be 'gelu' or 'relu'")
        self.ffn = nn.Sequential(
            nn.Linear(int(embed_dims), int(embed_dims) * int(ffn_ratio)),
            act,
            nn.Dropout(float(dropout)),
            nn.Linear(int(embed_dims) * int(ffn_ratio), int(embed_dims)),
        )
        self.norm2 = nn.LayerNorm(int(embed_dims))
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)
            if not key_padding_mask.any():
                key_padding_mask = None
        attn_out, _ = self.attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = self.norm1(tokens + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ffn(x)))
