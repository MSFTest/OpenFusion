# -*- coding: utf-8 -*-
"""
pointpainting_camera_to_image_projection

说明：
    PointPainting 私有补全算子。该算子复用 fusionrepo 通用 3D 点投影算子，
    将相机坐标系点云投影到图像平面像素坐标。

Args:
    points_camera:
        相机坐标系点云，shape 为 (N, 4)，前 3 维为 (x, y, z)。

    P:
        相机投影矩阵，shape 为 (4, 4) 或 (3, 4)。

    R0_rect:
        相机校正矩阵，shape 为 (4, 4) 或 (3, 3)。

    eps:
        深度归一化时的最小值，避免除零。

Returns:
    pixel_coords:
        图像平面像素坐标，shape 为 (N, 2)，格式为 (u, v)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _pointpainting_camera_to_image_projection_impl(
    points_camera: torch.Tensor,
    P: torch.Tensor,
    R0_rect: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if points_camera.dim() != 2 or points_camera.shape[1] < 3:
        raise ValueError("points_camera must have shape (N, C>=3)")

    if P.shape[-2:] == (4, 4):
        p34 = P[:3, :4]
    elif P.shape[-2:] == (3, 4):
        p34 = P
    else:
        raise ValueError("P must have shape (4, 4) or (3, 4)")

    if R0_rect.shape[-2:] == (4, 4):
        r44 = R0_rect
    elif R0_rect.shape[-2:] == (3, 3):
        r44 = torch.eye(4, device=R0_rect.device, dtype=R0_rect.dtype)
        r44[:3, :3] = R0_rect
    else:
        raise ValueError("R0_rect must have shape (4, 4) or (3, 3)")

    proj_matrix = p34.to(device=points_camera.device, dtype=points_camera.dtype) @ r44.to(
        device=points_camera.device,
        dtype=points_camera.dtype,
    )
    return torch.ops.fusionrepo.points_to_image(points_camera[:, :3], proj_matrix, float(eps))


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pointpainting_camera_to_image_projection", _pointpainting_camera_to_image_projection_impl)


def _pointpainting_camera_to_image_projection_meta(
    points_camera: torch.Tensor,
    P: torch.Tensor,
    R0_rect: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty((points_camera.shape[0], 2), device="meta", dtype=points_camera.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pointpainting_camera_to_image_projection", _pointpainting_camera_to_image_projection_meta)


def pointpainting_camera_to_image_projection(
    points_camera: torch.Tensor,
    P: torch.Tensor,
    R0_rect: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pointpainting_camera_to_image_projection(
        points_camera,
        P,
        R0_rect,
        float(eps),
    )
