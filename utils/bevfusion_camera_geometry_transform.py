# -*- coding: utf-8 -*-
"""
BEVfusion 相机几何刚体变换 wrapper。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _make_transform(R: torch.Tensor, t: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = R.to(device=device, dtype=dtype)
    transform[:3, 3] = t.to(device=device, dtype=dtype)
    return transform


def _bevfusion_camera_geometry_transform_impl(
    points_3d: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        points_3d: (D, H, W, 3)
            相机坐标系或中间坐标系下的三维点。
        R: (3, 3)
            从输入坐标系到目标坐标系的旋转矩阵。
        t: (3,)
            从输入坐标系到目标坐标系的平移向量。

    Returns:
        transformed_points: (D, H, W, 3)
            目标坐标系下的三维点。
    """
    if points_3d.dim() != 4 or points_3d.shape[-1] != 3:
        raise ValueError("points_3d must have shape (D, H, W, 3)")
    if R.shape != (3, 3):
        raise ValueError("R must have shape (3, 3)")
    if t.shape != (3,):
        raise ValueError("t must have shape (3,)")

    d, h, w, _ = points_3d.shape
    transform = _make_transform(R, t, points_3d.dtype, points_3d.device)
    transformed = torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        points_3d.reshape(-1, 3),
        transform,
        False,
        1e-6,
    )
    return transformed.reshape(d, h, w, 3)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "bevfusion_camera_geometry_transform",
    _bevfusion_camera_geometry_transform_impl,
)


def _bevfusion_camera_geometry_transform_meta(
    points_3d: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(tuple(points_3d.shape), device="meta", dtype=points_3d.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "bevfusion_camera_geometry_transform",
    _bevfusion_camera_geometry_transform_meta,
)


def bevfusion_camera_geometry_transform(
    points_3d: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.bevfusion_camera_geometry_transform(points_3d, R, t)
