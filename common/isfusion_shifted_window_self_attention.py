# -*- coding: utf-8 -*-
"""
IS-Fusion G2R shifted-window self-attention 融合块。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.token_self_attention_ffn_block import TokenSelfAttentionFFNBlock


class ISFusionShiftedWindowSelfAttentionBlock(nn.Module):
    """
    串联 non-shift 与 shifted window self-attention。

    Args:
        embed_dims: token 通道数。
        num_heads: 多头注意力 head 数。
        ffn_ratio: FFN 隐藏层相对通道倍率。
        dropout: dropout 比例。

    Forward Args:
        window_tokens: 普通窗口 token，shape 为 (num_windows, T, C)。
        window_mask: 普通窗口 padding mask，shape 为 (num_windows, T)。
        shifted_window_tokens: shifted 窗口 token，shape 为 (num_windows_shift, T, C)。
        shifted_window_mask: shifted 窗口 padding mask，shape 为 (num_windows_shift, T)。

    Returns:
        tuple[Tensor, Tensor]: 普通窗口增强 token 与 shifted 窗口增强 token。
    """

    def __init__(self, embed_dims: int = 128, num_heads: int = 8, ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.non_shift_block = TokenSelfAttentionFFNBlock(embed_dims, num_heads, ffn_ratio, dropout)
        self.shift_block = TokenSelfAttentionFFNBlock(embed_dims, num_heads, ffn_ratio, dropout)
        self.bridge = nn.Linear(int(embed_dims), int(embed_dims))

    def forward(
        self,
        window_tokens: torch.Tensor,
        window_mask: torch.Tensor = None,
        shifted_window_tokens: torch.Tensor = None,
        shifted_window_mask: torch.Tensor = None,
    ):
        enhanced = self.non_shift_block(window_tokens, window_mask)
        if shifted_window_tokens is None:
            shifted_window_tokens = enhanced
        shifted_seed = shifted_window_tokens + self.bridge(shifted_window_tokens)
        shifted = self.shift_block(shifted_seed, shifted_window_mask)
        return enhanced, shifted
