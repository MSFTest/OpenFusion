# -*- coding: utf-8 -*-
"""
topk_after_nms

说明：
    将 NMS 后的 top-k 截断注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.topk_after_nms(keep_indices, top_k)

兼容性：
    - 兼容 PyTorch 2.0.1
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    对 NMS 输出的保留索引做截断，只保留前 top_k 个。

输入：
    keep_indices: (N,)
        NMS 后保留下来的索引，通常已经按分数从高到低的顺序排列

    top_k: int
        最终保留数量上限

输出：
    selected_indices: (K,)
        截断后的索引
"""

from __future__ import annotations

import torch
from torch.library import Library




# ============================================================
# 2) Python 实现
# ============================================================
def _topk_after_nms_impl(
    keep_indices: torch.Tensor,
    top_k: int
) -> torch.Tensor:
    if keep_indices.dim() != 1:
        raise ValueError("keep_indices must have shape (N,)")

    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    k = min(int(top_k), int(keep_indices.numel()))
    return keep_indices[:k].to(dtype=torch.int64)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("topk_after_nms", _topk_after_nms_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _topk_after_nms_meta(
    keep_indices: torch.Tensor,
    top_k: int
) -> torch.Tensor:
    k = min(int(top_k), int(keep_indices.shape[0]))
    return torch.empty((k,), device="meta", dtype=torch.int64)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("topk_after_nms", _topk_after_nms_meta)


# ============================================================
# 4) Python 包装
# ============================================================
def topk_after_nms(
    keep_indices: torch.Tensor,
    top_k: int
) -> torch.Tensor:
    return torch.ops.fusionrepo.topk_after_nms(
        keep_indices,
        int(top_k)
    )