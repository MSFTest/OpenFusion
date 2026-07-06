# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有远距离目标判定算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_object_decision_impl(
    centroids: torch.Tensor,
    labels: torch.Tensor,
    depth_thresholds: torch.Tensor,
) -> torch.Tensor:
    """依据类别距离阈值判定远距离实例。

    Args:
        centroids: ``(K, 3)``，相机坐标系三维质心 ``(x, y, z)``。
        labels: ``(K,)``，类别索引。
        depth_thresholds: ``(C,)``，每个类别的深度阈值。

    Returns:
        is_faraway: ``(K,)`` 布尔张量，严格大于阈值时为 True。
    """
    if centroids.dim() != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must have shape (K, 3)")
    if labels.dim() != 1 or labels.shape[0] != centroids.shape[0]:
        raise ValueError("labels must have shape (K,)")
    if depth_thresholds.dim() != 1:
        raise ValueError("depth_thresholds must have shape (C,)")
    return centroids[:, 2] > depth_thresholds[labels]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("faraway_object_decision", _faraway_object_decision_impl)


def _faraway_object_decision_meta(
    centroids: torch.Tensor,
    labels: torch.Tensor,
    depth_thresholds: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((centroids.shape[0],), device="meta", dtype=torch.bool)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("faraway_object_decision", _faraway_object_decision_meta)


def faraway_object_decision(
    centroids: torch.Tensor,
    labels: torch.Tensor,
    depth_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_object_decision(
        centroids, labels, depth_thresholds
    )
