# -*- coding: utf-8 -*-
"""
multihead_attention_fusion

说明：
    该文件实现 Multi-Head Attention 原语，
    供 operator_registry.py 中注册后的

        torch.ops.fusionrepo.multihead_attention_fusion

    调用。

功能：
    执行标准 Multi-Head Attention：

        softmax(QKᵀ / √d) V

    支持：
        - Self Attention
        - Cross Attention

    不包含：
        - FFN
        - Residual
        - LayerNorm

    这些由网络结构层负责。

输入：

    query : (B , Nq , C)
    key   : (B , Nk , C)
    value : (B , Nk , C)

    num_heads : int

输出：

    out : (B , Nq , C)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _multihead_attention_fusion_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int):

    B, Nq, C = query.shape
    Nk = key.shape[1]

    if C % num_heads != 0:
        raise ValueError("channels must be divisible by num_heads")

    head_dim = C // num_heads

    # reshape
    q = query.view(B, Nq, num_heads, head_dim).transpose(1, 2)
    k = key.view(B, Nk, num_heads, head_dim).transpose(1, 2)
    v = value.view(B, Nk, num_heads, head_dim).transpose(1, 2)

    # attention score
    attn = torch.matmul(q, k.transpose(-2, -1))
    attn = attn / (head_dim ** 0.5)

    attn = F.softmax(attn, dim=-1)

    # attention output
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(B, Nq, C)

    return out


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "multihead_attention_fusion",
    _multihead_attention_fusion_impl
)


# ============================================================
# Meta
# ============================================================

def _multihead_attention_fusion_meta(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int):

    B, Nq, C = query.shape

    return torch.empty(
        (B, Nq, C),
        device="meta",
        dtype=query.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "multihead_attention_fusion",
    _multihead_attention_fusion_meta
)


# ============================================================
# wrapper
# ============================================================

def multihead_attention_fusion(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int):

    return torch.ops.fusionrepo.multihead_attention_fusion(
        query,
        key,
        value,
        int(num_heads)
    )