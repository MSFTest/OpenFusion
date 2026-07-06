# -*- coding: utf-8 -*-
"""
generate_rois

说明：
    该文件实现“proposal -> ROI”通用函数算子，
    供 operator_registry.py 中注册后的

        torch.ops.fusionrepo.generate_rois

    调用。

功能：
    根据 proposal 框与 batch 索引，
    构建统一的 ROI 表示：

        [batch_idx, x, y, z, dx, dy, dz, yaw]

输入：
    proposals : Tensor
        shape 为 (N, 7)
        每个 proposal 格式为：
        [x, y, z, dx, dy, dz, yaw]

    batch_indices : Tensor
        shape 为 (N,)
        每个 proposal 对应的 batch 索引

输出：
    rois : Tensor
        shape 为 (N, 8)
        格式为：
        [batch_idx, x, y, z, dx, dy, dz, yaw]
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _generate_rois_impl(
        proposals: torch.Tensor,
        batch_indices: torch.Tensor) -> torch.Tensor:
    if proposals.dim() != 2 or proposals.size(1) != 7:
        raise ValueError("proposals must have shape (N, 7)")

    if batch_indices.dim() != 1 or batch_indices.size(0) != proposals.size(0):
        raise ValueError("batch_indices must have shape (N,)")

    batch_indices = (
        batch_indices.to(device=proposals.device, dtype=proposals.dtype)
        .unsqueeze(1)
    )

    rois = torch.cat([batch_indices, proposals], dim=1)
    return rois


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "generate_rois",
    _generate_rois_impl
)


# ============================================================
# Meta
# ============================================================

def _generate_rois_meta(
        proposals: torch.Tensor,
        batch_indices: torch.Tensor) -> torch.Tensor:
    if proposals.dim() != 2 or proposals.size(1) != 7:
        raise ValueError("proposals must have shape (N, 7)")

    n = proposals.shape[0]

    return torch.empty(
        (n, 8),
        device="meta",
        dtype=proposals.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "generate_rois",
    _generate_rois_meta
)


# ============================================================
# wrapper
# ============================================================

def generate_rois(
        proposals: torch.Tensor,
        batch_indices: torch.Tensor) -> torch.Tensor:
    return torch.ops.fusionrepo.generate_rois(
        proposals,
        batch_indices
    )