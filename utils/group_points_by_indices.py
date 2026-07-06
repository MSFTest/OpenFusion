# -*- coding: utf-8 -*-
"""
group_points_by_indices

说明：
    根据整数 group id 将点或点特征打包为稠密分组张量。
    该算子只负责按已给定的 group_indices 分组、截断与补零，不负责生成 group id。

Args:
    points: (N, C) 或 (B, N, C)
        输入点或点特征。

    group_indices: (N,) 或 (B, N)
        每个点所属的 group id。小于 0 的点视为无效点。

    num_groups: int
        分组总数 G。

    max_points_per_group: int
        每组最多保留点数 M。

    random_sample: bool
        当组内点数超过 M 时，是否随机采样；False 时保留前 M 个点。

Returns:
    grouped_points: (G, M, C) 或 (B, G, M, C)
        分组后的稠密点张量。

    grouped_mask: (G, M) 或 (B, G, M)
        有效点掩码。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _group_points_by_indices_impl(
    points: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
    max_points_per_group: int,
    random_sample: bool = False,
):
    single_batch = points.dim() == 2
    if single_batch:
        points = points.unsqueeze(0)
        group_indices = group_indices.unsqueeze(0)

    if points.dim() != 3:
        raise ValueError("points must have shape (N, C) or (B, N, C)")
    if group_indices.dim() != 2:
        raise ValueError("group_indices must have shape (N,) or (B, N)")
    if points.shape[:2] != group_indices.shape:
        raise ValueError("points and group_indices must share batch and point dimensions")

    B, _, C = points.shape
    G = int(num_groups)
    M = int(max_points_per_group)
    grouped_points = torch.zeros((B, G, M, C), device=points.device, dtype=points.dtype)
    grouped_mask = torch.zeros((B, G, M), device=points.device, dtype=torch.bool)

    for b in range(B):
        for g in range(G):
            indices = torch.nonzero(group_indices[b] == g, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                continue
            if indices.numel() > M:
                if random_sample:
                    perm = torch.randperm(indices.numel(), device=points.device)[:M]
                    indices = indices[perm]
                else:
                    indices = indices[:M]
            count = indices.numel()
            grouped_points[b, g, :count] = points[b, indices]
            grouped_mask[b, g, :count] = True

    if single_batch:
        grouped_points = grouped_points.squeeze(0)
        grouped_mask = grouped_mask.squeeze(0)
    return grouped_points, grouped_mask


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("group_points_by_indices", _group_points_by_indices_impl)


def _group_points_by_indices_meta(
    points: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
    max_points_per_group: int,
    random_sample: bool = False,
):
    G = int(num_groups)
    M = int(max_points_per_group)
    C = points.shape[-1]
    if points.dim() == 2:
        return (
            torch.empty((G, M, C), device="meta", dtype=points.dtype),
            torch.empty((G, M), device="meta", dtype=torch.bool),
        )
    B = points.shape[0]
    return (
        torch.empty((B, G, M, C), device="meta", dtype=points.dtype),
        torch.empty((B, G, M), device="meta", dtype=torch.bool),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("group_points_by_indices", _group_points_by_indices_meta)


def group_points_by_indices(
    points: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
    max_points_per_group: int,
    random_sample: bool = False,
):
    return torch.ops.fusionrepo.group_points_by_indices(
        points,
        group_indices,
        int(num_groups),
        int(max_points_per_group),
        bool(random_sample),
    )
