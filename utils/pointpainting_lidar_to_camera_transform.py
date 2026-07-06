# -*- coding: utf-8 -*-
"""
pointpainting_lidar_to_camera_transform

说明：
    PointPainting 私有补全算子。该算子复用 fusionrepo 通用齐次变换算子，
    将 LiDAR 坐标系点云变换到相机坐标系，并保留原始反射强度。

Args:
    points_lidar:
        LiDAR 点云，shape 为 (N, 4)，格式为 (x, y, z, intensity)。

    transform:
        LiDAR 到相机坐标系的 4x4 齐次变换矩阵，shape 为 (4, 4)。

Returns:
    points_camera:
        相机坐标系点云，shape 为 (N, 4)，格式为
        (x_cam, y_cam, z_cam, intensity)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _pointpainting_lidar_to_camera_transform_impl(
    points_lidar: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    if points_lidar.dim() != 2 or points_lidar.shape[1] < 4:
        raise ValueError("points_lidar must have shape (N, 4) or (N, C>=4)")
    xyz_camera = torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        points_lidar[:, :3],
        transform,
        False,
        1e-6,
    ).squeeze(0)
    return torch.cat([xyz_camera, points_lidar[:, 3:4]], dim=1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pointpainting_lidar_to_camera_transform", _pointpainting_lidar_to_camera_transform_impl)


def _pointpainting_lidar_to_camera_transform_meta(
    points_lidar: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((points_lidar.shape[0], 4), device="meta", dtype=points_lidar.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pointpainting_lidar_to_camera_transform", _pointpainting_lidar_to_camera_transform_meta)


def pointpainting_lidar_to_camera_transform(
    points_lidar: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pointpainting_lidar_to_camera_transform(points_lidar, transform)
