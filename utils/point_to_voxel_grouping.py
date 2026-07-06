# -*- coding: utf-8 -*-
"""
point_to_voxel_grouping

说明：
    该文件实现"点到体素网格分组"原语，
    供 operator_registry.py 中注册后的
    torch.ops.fusionrepo.point_to_voxel_grouping 调用。

功能：
    将 3D 点云根据体素网格进行分组离散化，输出：
    - 每个体素的点集合
    - 体素坐标
    - 每个体素的点数统计

    该算子只负责：
    - 范围过滤
    - 坐标量化
    - 分组统计
    - 点数截断

    不负责：
    - 体素内特征聚合（由调用方自行处理）
    - 坐标变换

输入：
    points: (N, C)
        输入点云张量，前 3 维为 (x, y, z)，其余为点特征。

    voxel_size: (3,)
        体素尺寸 [vx, vy, vz]。

    point_cloud_range: (6,)
        点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。

    max_points_per_voxel: int
        每个体素最大保留点数。

    max_voxels: int
        最大体素数量。

输出：
    voxels: (M, T, C)
        每个体素的点集，M 为体素数，T 为 max_points_per_voxel。
        未填满位置为 0。

    coords: (M, 3)
        每个体素的整数坐标索引 (ix, iy, iz)。

    num_points_per_voxel: (M,)
        每个体素的真实点数。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _point_to_voxel_grouping_impl(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_points_per_voxel: int,
    max_voxels: int
):
    if points.ndim != 2:
        raise ValueError(f"points must be 2D (N,C), got {tuple(points.shape)}")
    if points.shape[1] < 3:
        raise ValueError("points must contain at least xyz coordinates")
    if voxel_size.shape != (3,):
        raise ValueError(f"voxel_size must be (3,), got {tuple(voxel_size.shape)}")
    if point_cloud_range.shape != (6,):
        raise ValueError(f"point_cloud_range must be (6,), got {tuple(point_cloud_range.shape)}")

    device = points.device
    dtype = points.dtype

    voxel_size = voxel_size.to(device=device, dtype=torch.float32)
    point_cloud_range = point_cloud_range.to(device=device, dtype=torch.float32)

    xyz = points[:, :3]
    feats = points

    pc_min = point_cloud_range[:3]
    pc_max = point_cloud_range[3:]

    mask = ((xyz >= pc_min) & (xyz < pc_max)).all(dim=1)
    xyz = xyz[mask]
    feats = feats[mask]

    if xyz.numel() == 0:
        M = 0
        T = max_points_per_voxel
        C = feats.shape[1] if feats.numel() > 0 else points.shape[1]
        empty_voxels = torch.empty((M, T, C), device=device, dtype=dtype)
        empty_coords = torch.empty((M, 3), device=device, dtype=torch.int32)
        empty_num = torch.empty((M,), device=device, dtype=torch.int32)
        return empty_voxels, empty_coords, empty_num

    voxel_coords = torch.floor((xyz - pc_min) / voxel_size).to(torch.int32)

    unique_coords, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)

    M = unique_coords.shape[0]
    T = max_points_per_voxel
    C = feats.shape[1]

    if M > max_voxels:
        unique_coords = unique_coords[:max_voxels]
        M = max_voxels

    voxels = torch.zeros((M, T, C), device=device, dtype=dtype)
    coords = unique_coords.clone()
    num_points_per_voxel = torch.zeros((M,), device=device, dtype=torch.int32)

    for i in range(M):
        mask_i = (inverse == i)
        pts_i = feats[mask_i]
        n_i = min(pts_i.shape[0], T)
        voxels[i, :n_i] = pts_i[:n_i]
        num_points_per_voxel[i] = n_i

    return voxels, coords, num_points_per_voxel


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("point_to_voxel_grouping", _point_to_voxel_grouping_impl)


# ============================================================
# Meta
# ============================================================

def _point_to_voxel_grouping_meta(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_points_per_voxel: int,
    max_voxels: int
):
    M = max_voxels
    T = max_points_per_voxel
    C = points.shape[1] if points.dim() == 2 and points.shape[0] > 0 else 0

    voxels = torch.empty((M, T, C), device="meta", dtype=points.dtype)
    coords = torch.empty((M, 3), device="meta", dtype=torch.int32)
    num_points = torch.empty((M,), device="meta", dtype=torch.int32)

    return voxels, coords, num_points


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("point_to_voxel_grouping", _point_to_voxel_grouping_meta)


# ============================================================
# Python wrapper
# ============================================================

def point_to_voxel_grouping(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_points_per_voxel: int = 32,
    max_voxels: int = 20000
):
    """
    Python wrapper：内部调用 torch.ops.fusionrepo.point_to_voxel_grouping

    Args:
        points: (N, C)
        voxel_size: (3,)
        point_cloud_range: (6,)
        max_points_per_voxel: int
        max_voxels: int

    Returns:
        voxels: (M, T, C)
        coords: (M, 3)
        num_points_per_voxel: (M,)
    """
    return torch.ops.fusionrepo.point_to_voxel_grouping(
        points,
        voxel_size,
        point_cloud_range,
        max_points_per_voxel,
        max_voxels
    )