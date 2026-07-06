# -*- coding: utf-8 -*-
"""
compute_voxel_centroid

说明：
    该文件实现 voxel 内点云质心计算原语，
    供 operator_registry.py 中注册后的：

        torch.ops.fusionrepo.compute_voxel_centroid

    调用。

功能：
    对每个 voxel 内的点云计算几何中心（centroid）：

        centroid = sum(points) / num_points

    仅执行几何平均计算，不包含：
        - 坐标变换
        - 特征编码
        - voxel 筛选
        - mask 处理

输入：
    voxel_points: (M, T, 3)
        voxel 内点坐标

    voxel_num_points: (M,)
        每个 voxel 内有效点数

输出：
    centroids: (M, 3)
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _compute_voxel_centroid_impl(
        voxel_points: torch.Tensor,
        voxel_num_points: torch.Tensor):

    points_sum = voxel_points.sum(dim=1)

    normalizer = torch.clamp_min(
        voxel_num_points.view(-1, 1).to(voxel_points.dtype),
        min=1.0
    )

    centroids = points_sum / normalizer

    return centroids


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "compute_voxel_centroid",
    _compute_voxel_centroid_impl
)


# ============================================================
# Meta
# ============================================================

def _compute_voxel_centroid_meta(
        voxel_points: torch.Tensor,
        voxel_num_points: torch.Tensor):

    M = voxel_points.shape[0]

    return torch.empty(
        (M, 3),
        device="meta",
        dtype=voxel_points.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "compute_voxel_centroid",
    _compute_voxel_centroid_meta
)


# ============================================================
# wrapper
# ============================================================

def compute_voxel_centroid(
        voxel_points: torch.Tensor,
        voxel_num_points: torch.Tensor):

    return torch.ops.fusionrepo.compute_voxel_centroid(
        voxel_points,
        voxel_num_points
    )