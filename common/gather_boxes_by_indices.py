# -*- coding: utf-8 -*-
"""
gather_boxes_by_indices

说明：
    该文件实现基于索引提取候选框的通用函数算子，
    用于在 NMS / Top-K 后根据保留索引提取最终 proposal 框。

功能：
    根据 keep_indices 从 boxes 中选取对应行。

输入：
    boxes : Tensor
        候选框张量，形状通常为 (N, D)

    keep_indices : Tensor
        保留索引，形状为 (K,)

输出：
    selected_boxes : Tensor
        选中的候选框，形状为 (K, D)
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _gather_boxes_by_indices_impl(
        boxes: torch.Tensor,
        keep_indices: torch.Tensor):
    if boxes.dim() != 2:
        raise ValueError("boxes must have shape (N, D)")

    if keep_indices.dim() != 1:
        raise ValueError("keep_indices must have shape (K,)")

    keep_indices = keep_indices.to(dtype=torch.long)

    return boxes[keep_indices]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("gather_boxes_by_indices", _gather_boxes_by_indices_impl)


# ============================================================
# Meta
# ============================================================

def _gather_boxes_by_indices_meta(
        boxes: torch.Tensor,
        keep_indices: torch.Tensor):
    k = keep_indices.shape[0]
    d = boxes.shape[1]

    return torch.empty(
        (k, d),
        device="meta",
        dtype=boxes.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("gather_boxes_by_indices", _gather_boxes_by_indices_meta)


# ============================================================
# Wrapper
# ============================================================

def gather_boxes_by_indices(
        boxes: torch.Tensor,
        keep_indices: torch.Tensor):
    return torch.ops.fusionrepo.gather_boxes_by_indices(
        boxes,
        keep_indices
    )