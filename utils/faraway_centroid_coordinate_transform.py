# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有外部质心坐标变换算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_centroid_coordinate_transform_impl(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """将分组点云平移到以外部质心为原点的局部坐标系。

    Args:
        clusters: ``(K, P, 3)``，对齐后的相机坐标系点云。
        counts: ``(K,)``，每组有效点数。
        centroids: ``(K, 3)``，外部估计的三维质心。

    Returns:
        local_clusters: ``(K, P, 3)``，局部坐标点云，padding 为零。
    """
    if clusters.dim() != 3 or clusters.shape[-1] != 3:
        raise ValueError("clusters must have shape (K, P, 3)")
    if counts.shape != (clusters.shape[0],):
        raise ValueError("counts must have shape (K,)")
    if centroids.shape != (clusters.shape[0], 3):
        raise ValueError("centroids must have shape (K, 3)")

    transforms = torch.eye(4, device=clusters.device, dtype=clusters.dtype).repeat(
        clusters.shape[0], 1, 1
    )
    transforms[:, :3, 3] = -centroids.to(dtype=clusters.dtype)
    local = torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        clusters, transforms, False, 1e-6
    )
    valid = torch.arange(clusters.shape[1], device=clusters.device).unsqueeze(0) < counts.unsqueeze(1)
    return local * valid.unsqueeze(-1).to(dtype=local.dtype)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "faraway_centroid_coordinate_transform",
    _faraway_centroid_coordinate_transform_impl,
)


def _faraway_centroid_coordinate_transform_meta(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(clusters, device="meta")


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "faraway_centroid_coordinate_transform",
    _faraway_centroid_coordinate_transform_meta,
)


def centroid_coordinate_transform(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_centroid_coordinate_transform(
        clusters, counts, centroids
    )
