# -*- coding: utf-8 -*-
"""
attention_output_projection

说明：
    该文件实现“注意力输出投影”层形式通用算子，
    用于补齐 HMFI 的 QFM_MultiHeadAttention 中最后的 out_proj。

功能：
    将 multihead_attention_fusion 的输出：
        - (1, N, C_attn)
    或
        - (N, C_attn)
    通过线性层映射为：
        - (N, out_channels)
    或
        - (1, N, out_channels)

说明：
    该模块属于可学习算子，带有线性投影参数；
    因此不放到 operator_registry.py 中做 DEF 注册，
    而是作为 block 模块直接实例化使用。

Args:
    in_channels: int
        注意力输出特征维度，通常为 d_v。

    out_channels: int
        输出特征维度，对应 HMFI 原模块中的 out_channels。

    bias: bool, optional
        线性层是否带偏置，默认 False，与 HMFI 原实现保持一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionOutputProjection(nn.Module):
    """
    注意力输出投影模块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False
    ):
        super().__init__()

        self.out_proj = nn.Linear(
            in_channels,
            out_channels,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:
                - (1, N, C_attn)
                - (N, C_attn)

        Returns:
            y:
                - 若输入为 (1, N, C_attn)，输出为 (N, out_channels)
                - 若输入为 (N, C_attn)，输出为 (N, out_channels)
        """
        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError("When x is 3D, expected shape (1, N, C_attn)")
            x = x.squeeze(0)  # (N, C_attn)

        if x.dim() != 2:
            raise ValueError("x must have shape (1, N, C_attn) or (N, C_attn)")

        y = self.out_proj(x)
        return y