# -*- coding: utf-8 -*-
"""
voxel_projection

说明：
    该文件实现"体素坐标到图像平面投影"的 VirConv 私有 wrapper 算子。

功能：
    第一步：将体素坐标转换为体素中心坐标
    第二步：调用已有的通用投影算子 points_to_image

    该算子只负责：
    - voxel_coords -> voxel_centers 转换
    - 调用已有的 points_to_image 算子

    不负责：
    - 投影矩阵构建

输入：
    voxel_coords: (M, 3)
        体素坐标索引 (ix, iy, iz)。

    voxel_size: (3,)
        体素尺寸 [vx, vy, vz]。

    point_cloud_range: (6,)
        点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]。

    proj_matrix: (3, 4)
        相机投影矩阵（K[R|t]）。

输出：
    pixel_coords: (M, 2)
        每个体素中心对应的图像平面像素坐标 (u, v)。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _voxel_projection_impl(
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    proj_matrix: torch.Tensor
):
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
        raise ValueError(f"voxel_coords must be (M,3), got {tuple(voxel_coords.shape)}")

    device = voxel_coords.device

    voxel_size = voxel_size.to(device=device, dtype=torch.float32)
    point_cloud_range = point_cloud_range.to(device=device, dtype=torch.float32)
    proj_matrix = proj_matrix.to(device=device, dtype=torch.float32)

    # 第一步：计算体素中心坐标
    pc_min = point_cloud_range[:3]
    centers = (voxel_coords.to(torch.float32) + 0.5) * voxel_size + pc_min

    # 第二步：调用已有的通用投影算子 points_to_image
    pixel_coords = torch.ops.fusionrepo.points_to_image(centers, proj_matrix)

    return pixel_coords.clone()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("voxel_projection", _voxel_projection_impl)


# ============================================================
# Meta
# ============================================================

def _voxel_projection_meta(
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    proj_matrix: torch.Tensor
):
    m = voxel_coords.shape[0]
    return torch.empty((m, 2), device="meta", dtype=torch.float32)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("voxel_projection", _voxel_projection_meta)


# ============================================================
# Python wrapper
# ============================================================

def voxel_projection(
    voxel_coords: torch.Tensor,
    voxel_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
    proj_matrix: torch.Tensor
):
    """
    Python wrapper：内部调用 torch.ops.fusionrepo.voxel_projection

    Args:
        voxel_coords: (M, 3)
        voxel_size: (3,)
        point_cloud_range: (6,)
        proj_matrix: (3, 4)

    Returns:
        pixel_coords: (M, 2)
    """
    return torch.ops.fusionrepo.voxel_projection(
        voxel_coords,
        voxel_size,
        point_cloud_range,
        proj_matrix
    )