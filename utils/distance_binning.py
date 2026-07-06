# -*- coding: utf-8 -*-
"""
distance_binning

说明：
    该文件实现"体素距离分桶"的 VirConv 私有算子。

功能：
    根据体素在空间中的距离，将体素划分到不同距离 bin 中。

    该算子只负责：
    - 计算体素中心到原点距离
    - 按 bin_edges 分桶

    不负责：
    - 投影
    - 特征变换

输入：
    voxel_coords: (M, 3)
        体素坐标索引 (ix, iy, iz)。

    voxel_size: (3,)
        体素尺寸 [vx, vy, vz]。

    point_cloud_range: (6,)
        点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。

    bin_edges: (K,)
        距离分桶边界（递增）。

输出：
    bin_ids: (M,)
        每个体素所属的距离 bin 编号，范围 [0, K]。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _distance_binning_impl(
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    bin_edges: torch.Tensor
):
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
        raise ValueError(f"voxel_coords must be (M,3), got {tuple(voxel_coords.shape)}")

    device = voxel_coords.device
    voxel_size = voxel_size.to(device=device, dtype=torch.float32)
    point_cloud_range = point_cloud_range.to(device=device, dtype=torch.float32)
    bin_edges = bin_edges.to(device=device, dtype=torch.float32)

    pc_min = point_cloud_range[:3]

    centers = (voxel_coords.to(torch.float32) + 0.5) * voxel_size + pc_min
    distances = torch.linalg.norm(centers, dim=1)

    bin_ids = torch.bucketize(distances, bin_edges)
    return bin_ids.clone()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("distance_binning", _distance_binning_impl)


# ============================================================
# Meta
# ============================================================

def _distance_binning_meta(
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    bin_edges: torch.Tensor
):
    m = voxel_coords.shape[0]
    return torch.empty((m,), device="meta", dtype=torch.int64)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("distance_binning", _distance_binning_meta)


# ============================================================
# Python wrapper
# ============================================================

def distance_binning(
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    bin_edges: torch.Tensor
):
    """
    Python wrapper：内部调用 torch.ops.fusionrepo.distance_binning

    Args:
        voxel_coords: (M, 3)
        voxel_size: (3,)
        point_cloud_range: (6,)
        bin_edges: (K,)

    Returns:
        bin_ids: (M,)
    """
    return torch.ops.fusionrepo.distance_binning(
        voxel_coords,
        voxel_size,
        point_cloud_range,
        bin_edges
    )