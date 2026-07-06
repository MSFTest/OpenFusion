# -*- coding: utf-8 -*-
"""
attention_tokens_to_batch

说明：
    该文件将“无 batch 的 token 形式注意力输入 -> 带 batch 的标准注意力输入”
    注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.attention_tokens_to_batch(x)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将 HMFI 中的 token 形式特征：
        - (N, C)
    转换为 multihead_attention_fusion 可直接处理的 batch 形式：
        - (1, N, C)

说明：
    该算子仅做形状适配，不改变数值含义，不引入可学习参数。

Args:
    x: (N, C)
        无 batch 的 token 特征

Returns:
    y: (1, N, C)
        带 batch 的注意力输入特征
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现
# ============================================================
def _attention_tokens_to_batch_impl(
    x: torch.Tensor
) -> torch.Tensor:
    """
    具体实现：
        (N, C) -> (1, N, C)
    """
    if x.dim() != 2:
        raise ValueError("x must have shape (N, C)")

    return x.unsqueeze(0)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("attention_tokens_to_batch", _attention_tokens_to_batch_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _attention_tokens_to_batch_meta(
    x: torch.Tensor
) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("x must have shape (N, C)")

    n, c = x.shape
    return torch.empty((1, n, c), device="meta", dtype=x.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("attention_tokens_to_batch", _attention_tokens_to_batch_meta)


# ============================================================
# 4) Python 包装
# ============================================================
def attention_tokens_to_batch(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.attention_tokens_to_batch
    """
    return torch.ops.fusionrepo.attention_tokens_to_batch(x)