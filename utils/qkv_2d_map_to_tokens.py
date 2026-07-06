# -*- coding: utf-8 -*-
"""
qkv_2d_map_to_tokens

说明：
    该文件将“1x1 Conv 输出的二维特征图 -> Q/K/V token 形式”
    注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.qkv_2d_map_to_tokens(query_map, key_map, value_map)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将 channel_reduce_1x1_2d 输出的 Q/K/V 二维特征图结果：
        - query_map: (1, d_k, M, 1) 或 (B, d_k, M, 1)
        - key_map:   (1, d_k, L, 1) 或 (B, d_k, L, 1)
        - value_map: (1, d_v, L, 1) 或 (B, d_v, L, 1)

    还原为 HMFI 注意力模块需要的 token 表示：
        - query: (M, d_k) 或 (B, M, d_k)
        - key:   (L, d_k) 或 (B, L, d_k)
        - value: (L, d_v) 或 (B, L, d_v)

说明：
    该算子只做形状恢复与结果组织，不引入可学习参数。

Args:
    query_map: (1, d_k, M, 1) 或 (B, d_k, M, 1)
    key_map:   (1, d_k, L, 1) 或 (B, d_k, L, 1)
    value_map: (1, d_v, L, 1) 或 (B, d_v, L, 1)

Returns:
    List[Tensor]:
        [query, key, value]
"""

from __future__ import annotations

from typing import List

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现
# ============================================================
def _restore_single_map(x: torch.Tensor) -> torch.Tensor:
    """
    - (1, C, N, 1) -> (N, C)
    - (B, C, N, 1) -> (B, N, C)
    """
    if x.dim() != 4:
        raise ValueError("input map must have shape (1, C, N, 1) or (B, C, N, 1)")
    if x.shape[-1] != 1:
        raise ValueError("last dimension of input map must be 1")

    y = x.squeeze(-1).transpose(1, 2)  # (B, N, C)

    if y.shape[0] == 1:
        return y.squeeze(0)            # (N, C)

    return y                           # (B, N, C)


def _qkv_2d_map_to_tokens_impl(
    query_map: torch.Tensor,
    key_map: torch.Tensor,
    value_map: torch.Tensor
) -> List[torch.Tensor]:
    """
    具体实现：
        query_map: (1, d_k, M, 1) 或 (B, d_k, M, 1)
        key_map:   (1, d_k, L, 1) 或 (B, d_k, L, 1)
        value_map: (1, d_v, L, 1) 或 (B, d_v, L, 1)

    返回：
        [query, key, value]
    """
    query = _restore_single_map(query_map)
    key = _restore_single_map(key_map)
    value = _restore_single_map(value_map)

    return [query, key, value]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("qkv_2d_map_to_tokens", _qkv_2d_map_to_tokens_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _restore_single_map_meta(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("input map must have shape (1, C, N, 1) or (B, C, N, 1)")
    if x.shape[-1] != 1:
        raise ValueError("last dimension of input map must be 1")

    b, c, n, _ = x.shape
    if b == 1:
        return torch.empty((n, c), device="meta", dtype=x.dtype)
    return torch.empty((b, n, c), device="meta", dtype=x.dtype)


def _qkv_2d_map_to_tokens_meta(
    query_map: torch.Tensor,
    key_map: torch.Tensor,
    value_map: torch.Tensor
) -> List[torch.Tensor]:
    query = _restore_single_map_meta(query_map)
    key = _restore_single_map_meta(key_map)
    value = _restore_single_map_meta(value_map)
    return [query, key, value]


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("qkv_2d_map_to_tokens", _qkv_2d_map_to_tokens_meta)


# ============================================================
# 4) Python 包装
# ============================================================
def qkv_2d_map_to_tokens(
    query_map: torch.Tensor,
    key_map: torch.Tensor,
    value_map: torch.Tensor
) -> List[torch.Tensor]:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.qkv_2d_map_to_tokens
    """
    return torch.ops.fusionrepo.qkv_2d_map_to_tokens(
        query_map,
        key_map,
        value_map
    )