# -*- coding: utf-8 -*-
"""
epnetpp_point_to_grid_scatter

说明：
    EPNet++ 点级特征到 BEV 网格的散射补全算子，注册为：
        torch.ops.fusionrepo.epnetpp_point_to_grid_scatter(point_features, point_indices, grid_h, grid_w)

    该算子保留 EPNet++ 原始 point_to_grid_scatter 的输入输出语义，将
    点级特征按 BEV 网格整数索引累加到 dense BEV 特征图。内部复用通用
    voxel_features_to_bev_scatter，并使用 reduce="sum" 保持同一网格多点累加。

Args:
    point_features: Tensor
        点级特征，shape 为 (B, N, C)。
    point_indices: Tensor
        点在 BEV 网格中的整数索引，shape 为 (B, N, 2)，顺序为 (y, x)。
        落在 [0, grid_h) 和 [0, grid_w) 外的点会被忽略。
    grid_h: int
        输出 BEV 网格高度。
    grid_w: int
        输出 BEV 网格宽度。

Returns:
    Tensor
        BEV 网格特征图，shape 为 (B, C, grid_h, grid_w)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnetpp_point_to_grid_scatter_impl(
    point_features: torch.Tensor,
    point_indices: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    if point_features.dim() != 3:
        raise ValueError("point_features must have shape (B, N, C)")
    if point_indices.dim() != 3 or point_indices.shape[-1] != 2:
        raise ValueError("point_indices must have shape (B, N, 2)")
    if point_features.shape[:2] != point_indices.shape[:2]:
        raise ValueError("point_features and point_indices must have matching B and N")

    batch_size, num_points, channels = point_features.shape
    features_flat = point_features.reshape(batch_size * num_points, channels)
    indices_flat = point_indices.reshape(batch_size * num_points, 2).to(
        device=point_features.device,
        dtype=torch.long,
    )
    batch_ids = torch.arange(batch_size, device=point_features.device, dtype=torch.long)
    batch_ids = batch_ids.view(batch_size, 1).expand(batch_size, num_points).reshape(-1, 1)
    zeros_z = torch.zeros_like(batch_ids)
    voxel_coords = torch.cat([batch_ids, zeros_z, indices_flat[:, 0:1], indices_flat[:, 1:2]], dim=1)

    return torch.ops.fusionrepo.voxel_features_to_bev_scatter(
        features_flat,
        voxel_coords,
        int(batch_size),
        int(grid_h),
        int(grid_w),
        "sum",
    ).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnetpp_point_to_grid_scatter", _epnetpp_point_to_grid_scatter_impl)


def _epnetpp_point_to_grid_scatter_meta(
    point_features: torch.Tensor,
    point_indices: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    batch_size, _, channels = point_features.shape
    return torch.empty(
        (batch_size, channels, int(grid_h), int(grid_w)),
        device="meta",
        dtype=point_features.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnetpp_point_to_grid_scatter", _epnetpp_point_to_grid_scatter_meta)


def epnetpp_point_to_grid_scatter(
    point_features: torch.Tensor,
    point_indices: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    return torch.ops.fusionrepo.epnetpp_point_to_grid_scatter(
        point_features,
        point_indices,
        int(grid_h),
        int(grid_w),
    )
