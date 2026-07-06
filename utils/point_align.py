# -*- coding: utf-8 -*-
"""
point_align

说明：
    该文件实现"点云特征对齐"的 VirConv 私有算子。

功能：
    对融合后的点云进行特征对齐处理，保证点特征格式统一。

    该算子只负责：
    - 返回输入的 clone

    不负责：
    - 特征变换

输入：
    points: (N, C)
        输入点云张量。

输出：
    points_aligned: (N, C)
        特征对齐后的点云张量（与输入形状一致）。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _point_align_impl(
    points: torch.Tensor
):
    if points.ndim != 2:
        raise ValueError(f"points must be 2D (N,C), got {tuple(points.shape)}")

    return points.clone()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("point_align", _point_align_impl)


# ============================================================
# Meta
# ============================================================

def _point_align_meta(
    points: torch.Tensor
):
    n, c = points.shape
    return torch.empty((n, c), device="meta", dtype=points.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("point_align", _point_align_meta)


# ============================================================
# Python wrapper
# ============================================================

def point_align(
    points: torch.Tensor
):
    """
    Python wrapper：内部调用 torch.ops.fusionrepo.point_align

    Args:
        points: (N, C)

    Returns:
        points_aligned: (N, C)
    """
    return torch.ops.fusionrepo.point_align(points)