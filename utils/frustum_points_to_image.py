# -*- coding: utf-8 -*-
"""
frustum_points_to_image

说明：
    F-ConvNet 风格的 frustum 点投影适配算子。
    该算子将相机内参 K 扩展为完整投影矩阵 [K|0]，复用
    fusionrepo::points_to_image 完成 3D 点到图像平面的投影，并同时返回
    点的深度值。

Args:
    points_3d: (N, 3) 或 (B, N, 3)
        相机坐标系或 rect 坐标系下的三维点，最后一维为 (x, y, z)。

    K: (3, 3) 或 (B, 3, 3)
        相机内参矩阵。

    eps: float
        投影除法中的深度下限，避免除 0。

Returns:
    pixel_coords: (N, 2) 或 (B, N, 2)
        图像平面像素坐标 (u, v)。

    depth: (N,) 或 (B, N)
        输入点的 z 方向深度。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _frustum_points_to_image_impl(
    points_3d: torch.Tensor,
    K: torch.Tensor,
    eps: float = 1e-6,
):
    if points_3d.shape[-1] != 3:
        raise ValueError("points_3d last dim must be 3")
    if K.shape[-2:] != (3, 3):
        raise ValueError("K must have shape (3, 3) or (B, 3, 3)")

    K = K.to(device=points_3d.device, dtype=points_3d.dtype)
    zeros_shape = K.shape[:-1] + (1,)
    zero_col = torch.zeros(zeros_shape, device=points_3d.device, dtype=points_3d.dtype)
    proj_matrix = torch.cat([K, zero_col], dim=-1)

    pixel_coords = torch.ops.fusionrepo.points_to_image(points_3d, proj_matrix, float(eps))
    depth = points_3d[..., 2].clone()
    return pixel_coords, depth


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("frustum_points_to_image", _frustum_points_to_image_impl)


def _frustum_points_to_image_meta(
    points_3d: torch.Tensor,
    K: torch.Tensor,
    eps: float = 1e-6,
):
    pixel_coords = torch.empty(
        points_3d.shape[:-1] + (2,),
        device="meta",
        dtype=points_3d.dtype,
    )
    depth = torch.empty(points_3d.shape[:-1], device="meta", dtype=points_3d.dtype)
    return pixel_coords, depth


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("frustum_points_to_image", _frustum_points_to_image_meta)


def frustum_points_to_image(
    points_3d: torch.Tensor,
    K: torch.Tensor,
    eps: float = 1e-6,
):
    return torch.ops.fusionrepo.frustum_points_to_image(points_3d, K, float(eps))
