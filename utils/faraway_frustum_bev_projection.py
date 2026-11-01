# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有点云 BEV 密度投影算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_frustum_bev_projection_impl(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    bev_height: int,
    bev_width: int,
    x_range: float,
    z_range: float,
) -> torch.Tensor:
    """将局部点云投影为 x-z 平面的单通道 BEV 密度图。

    Args:
        clusters: ``(K, P, 3)``，目标局部坐标系点云。
        counts: ``(K,)``，每组有效点数。
        bev_height: BEV 的 z 轴网格数 H。
        bev_width: BEV 的 x 轴网格数 W。
        x_range: x 轴覆盖 ``[-x_range, x_range]``。
        z_range: z 轴覆盖 ``[-z_range, z_range]``。

    Returns:
        bev_maps: ``(K, 1, H, W)``，每格数值为落入点数。
    """
    if clusters.dim() != 3 or clusters.shape[-1] != 3:
        raise ValueError("clusters must have shape (K, P, 3)")
    if counts.shape != (clusters.shape[0],):
        raise ValueError("counts must have shape (K,)")
    if bev_height <= 0 or bev_width <= 0 or x_range <= 0 or z_range <= 0:
        raise ValueError("BEV sizes and coordinate ranges must be positive")

    group_count, max_points, _ = clusters.shape
    valid = torch.arange(max_points, device=clusters.device).unsqueeze(0) < counts.unsqueeze(1)
    valid = (
        valid
        & (clusters[..., 0] >= -float(x_range))
        & (clusters[..., 0] < float(x_range))
        & (clusters[..., 2] >= -float(z_range))
        & (clusters[..., 2] < float(z_range))
    )
    batch_indices = torch.arange(group_count, device=clusters.device).unsqueeze(1).expand(-1, max_points)
    x_indices = torch.clamp(
        (((clusters[..., 0] + float(x_range)) / (2.0 * float(x_range))) * bev_width).long(),
        0,
        bev_width - 1,
    )
    z_indices = torch.clamp(
        (((clusters[..., 2] + float(z_range)) / (2.0 * float(z_range))) * bev_height).long(),
        0,
        bev_height - 1,
    )
    coordinates = torch.stack(
        [
            batch_indices[valid],
            torch.zeros_like(batch_indices[valid]),
            z_indices[valid],
            x_indices[valid],
        ],
        dim=1,
    )
    features = clusters.new_ones((coordinates.shape[0], 1))
    return torch.ops.fusionrepo.voxel_features_to_bev_scatter(
        features, coordinates, int(group_count), int(bev_height), int(bev_width), "sum"
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("faraway_frustum_bev_projection", _faraway_frustum_bev_projection_impl)


def _faraway_frustum_bev_projection_meta(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    bev_height: int,
    bev_width: int,
    x_range: float,
    z_range: float,
) -> torch.Tensor:
    return torch.empty(
        (clusters.shape[0], 1, int(bev_height), int(bev_width)),
        device="meta",
        dtype=clusters.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("faraway_frustum_bev_projection", _faraway_frustum_bev_projection_meta)


def frustum_bev_projection(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    bev_height: int,
    bev_width: int,
    x_range: float,
    z_range: float,
) -> torch.Tensor:
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_frustum_bev_projection(
        clusters,
        counts,
        int(bev_height),
        int(bev_width),
        float(x_range),
        float(z_range),
    )
