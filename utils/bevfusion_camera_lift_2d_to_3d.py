# -*- coding: utf-8 -*-
"""
BEVfusion 像素-深度反投影到三维空间 wrapper。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _bevfusion_camera_lift_2d_to_3d_impl(
    pixels_2d: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        pixels_2d: (H, W, 2)
            图像平面像素坐标，最后一维为 (u, v)。
        depths: (D,)
            离散深度 bin。
        K: (3, 3)
            相机内参矩阵。
        R: (3, 3)
            相机坐标系到目标坐标系的旋转矩阵。
        t: (3,)
            相机坐标系到目标坐标系的平移向量。

    Returns:
        points_3d: (D, H, W, 3)
            目标坐标系下的三维点。
    """
    if pixels_2d.dim() != 3 or pixels_2d.shape[-1] != 2:
        raise ValueError("pixels_2d must have shape (H, W, 2)")
    if depths.dim() != 1:
        raise ValueError("depths must have shape (D,)")

    h, w, _ = pixels_2d.shape
    d = depths.shape[0]
    pixels = pixels_2d.unsqueeze(0).expand(d, h, w, 2).contiguous()
    depth_grid = depths.to(device=pixels_2d.device, dtype=pixels_2d.dtype).view(d, 1, 1).expand(d, h, w)
    points_cam = torch.ops.fusionrepo.pixels_depth_to_camera_points(pixels, depth_grid, K)
    return torch.ops.fusionrepo.bevfusion_camera_geometry_transform(points_cam, R, t)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "bevfusion_camera_lift_2d_to_3d",
    _bevfusion_camera_lift_2d_to_3d_impl,
)


def _bevfusion_camera_lift_2d_to_3d_meta(
    pixels_2d: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (depths.shape[0], pixels_2d.shape[0], pixels_2d.shape[1], 3),
        device="meta",
        dtype=pixels_2d.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "bevfusion_camera_lift_2d_to_3d",
    _bevfusion_camera_lift_2d_to_3d_meta,
)


def bevfusion_camera_lift_2d_to_3d(
    pixels_2d: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.bevfusion_camera_lift_2d_to_3d(
        pixels_2d,
        depths,
        K,
        R,
        t,
    )
