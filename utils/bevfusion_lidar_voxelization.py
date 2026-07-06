# -*- coding: utf-8 -*-
"""
BEVfusion 批量 LiDAR 点云体素化 wrapper。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _bevfusion_lidar_voxelization_impl(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_points_per_voxel: int,
    max_voxels: int,
):
    """
    Args:
        points: (B, N, C)
            批量 LiDAR 点云，前三维为 (x, y, z)。
        voxel_size: (3,)
            体素尺寸 (vx, vy, vz)。
        point_cloud_range: (6,)
            点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。
        max_points_per_voxel: int
            每个体素最多保留点数。
        max_voxels: int
            每帧最多保留体素数。

    Returns:
        voxels: (M, T, C)
            批量拼接后的体素内点特征。
        coords: (M, 4)
            体素坐标，格式为 [batch_idx, z, y, x]。
        num_points_per_voxel: (M,)
            每个体素真实点数。
    """
    if points.dim() != 3:
        raise ValueError("points must have shape (B, N, C)")

    voxel_list = []
    coord_list = []
    num_list = []
    batch_size = points.shape[0]
    for batch_idx in range(batch_size):
        voxels, coords_xyz, num_points = torch.ops.fusionrepo.point_to_voxel_grouping(
            points[batch_idx],
            voxel_size,
            point_cloud_range,
            int(max_points_per_voxel),
            int(max_voxels),
        )
        if coords_xyz.numel() == 0:
            continue
        batch_col = torch.full(
            (coords_xyz.shape[0], 1),
            batch_idx,
            device=coords_xyz.device,
            dtype=torch.int32,
        )
        coords_zyx = torch.stack(
            [coords_xyz[:, 2], coords_xyz[:, 1], coords_xyz[:, 0]],
            dim=1,
        ).to(dtype=torch.int32)
        voxel_list.append(voxels)
        coord_list.append(torch.cat([batch_col, coords_zyx], dim=1))
        num_list.append(num_points)

    if not voxel_list:
        return (
            points.new_empty((0, int(max_points_per_voxel), points.shape[2])),
            torch.empty((0, 4), device=points.device, dtype=torch.int32),
            torch.empty((0,), device=points.device, dtype=torch.int32),
        )

    return (
        torch.cat(voxel_list, dim=0),
        torch.cat(coord_list, dim=0),
        torch.cat(num_list, dim=0),
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("bevfusion_lidar_voxelization", _bevfusion_lidar_voxelization_impl)


def _bevfusion_lidar_voxelization_meta(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_points_per_voxel: int,
    max_voxels: int,
):
    max_total = points.shape[0] * int(max_voxels)
    return (
        torch.empty(
            (max_total, int(max_points_per_voxel), points.shape[2]),
            device="meta",
            dtype=points.dtype,
        ),
        torch.empty((max_total, 4), device="meta", dtype=torch.int32),
        torch.empty((max_total,), device="meta", dtype=torch.int32),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("bevfusion_lidar_voxelization", _bevfusion_lidar_voxelization_meta)


def bevfusion_lidar_voxelization(
    points: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    max_points_per_voxel: int = 32,
    max_voxels: int = 20000,
):
    return torch.ops.fusionrepo.bevfusion_lidar_voxelization(
        points,
        voxel_size,
        point_cloud_range,
        int(max_points_per_voxel),
        int(max_voxels),
    )
