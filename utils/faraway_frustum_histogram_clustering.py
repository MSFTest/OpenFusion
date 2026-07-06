# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有逐轴直方图质心估计算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_frustum_histogram_clustering_impl(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    bins: int = 20,
) -> torch.Tensor:
    """估计每组 Frustum 点云的逐轴主峰中心。

    Args:
        clusters: ``(K, P, 3)``，相机坐标系中的分组点云。
        counts: ``(K,)``，每组有效点数。
        bins: 每个坐标轴的直方图区间数。

    Returns:
        centroids: ``(K, 3)``，逐轴最大频次区间的中心坐标。
    """
    if clusters.dim() != 3 or clusters.shape[-1] != 3:
        raise ValueError("clusters must have shape (K, P, 3)")
    if counts.dim() != 1 or counts.shape[0] != clusters.shape[0]:
        raise ValueError("counts must have shape (K,)")
    if bins <= 0:
        raise ValueError("bins must be positive")

    centroids = clusters.new_zeros((clusters.shape[0], 3))
    for index in range(int(clusters.shape[0])):
        count = int(counts[index].item())
        if count <= 0:
            continue
        points = clusters[index, :count, :]
        for dimension in range(3):
            values = points[:, dimension]
            value_min = torch.min(values)
            value_max = torch.max(values)
            if (value_max - value_min).abs() < 1e-6:
                centroids[index, dimension] = value_min
                continue
            histogram = torch.histc(
                values,
                bins=int(bins),
                min=float(value_min),
                max=float(value_max),
            )
            max_bin = int(torch.argmax(histogram).item())
            bin_width = (value_max - value_min) / float(bins)
            centroids[index, dimension] = value_min + (max_bin + 0.5) * bin_width
    return centroids


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "faraway_frustum_histogram_clustering",
    _faraway_frustum_histogram_clustering_impl,
)


def _faraway_frustum_histogram_clustering_meta(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    bins: int = 20,
) -> torch.Tensor:
    return torch.empty((clusters.shape[0], 3), device="meta", dtype=clusters.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "faraway_frustum_histogram_clustering",
    _faraway_frustum_histogram_clustering_meta,
)


def frustum_histogram_clustering(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    bins: int = 20,
) -> torch.Tensor:
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_frustum_histogram_clustering(
        clusters, counts, int(bins)
    )
