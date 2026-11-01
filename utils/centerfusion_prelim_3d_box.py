# -*- coding: utf-8 -*-
"""
centerfusion_prelim_3d_box

说明：
    CenterFusion 私有三维框初步构建算子。该算子先复用
    fusionrepo::pixels_depth_to_camera_points 将 2D 中心与深度反投影到
    相机坐标系，再拼接三维尺寸与 yaw，形成 CenterFusion 的初步 3D 框。

Args:
    centers_2d: (N, 2)
        图像或特征图坐标系中的中心点 (u, v)。
    depths: (N, 1)
        目标深度。
    dims_3d: (N, 3)
        三维尺寸 (length, width, height)。
    rots: (N, 1)
        目标 yaw。
    K: (3, 3)
        相机内参矩阵。

Returns:
    box3d: (N, 7)
        初步三维框，格式为 (x, y, z, length, width, height, yaw)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _centerfusion_prelim_3d_box_impl(
    centers_2d: torch.Tensor,
    depths: torch.Tensor,
    dims_3d: torch.Tensor,
    rots: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    if centers_2d.dim() != 2 or centers_2d.shape[1] != 2:
        raise ValueError("centers_2d must have shape (N, 2)")
    if depths.dim() != 2 or depths.shape[1] != 1:
        raise ValueError("depths must have shape (N, 1)")
    if dims_3d.dim() != 2 or dims_3d.shape[1] != 3:
        raise ValueError("dims_3d must have shape (N, 3)")
    if rots.dim() != 2 or rots.shape[1] != 1:
        raise ValueError("rots must have shape (N, 1)")

    points_cam = torch.ops.fusionrepo.pixels_depth_to_camera_points(
        centers_2d,
        depths,
        K,
    )
    return torch.cat([points_cam, dims_3d, rots], dim=1)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("centerfusion_prelim_3d_box", _centerfusion_prelim_3d_box_impl)


def _centerfusion_prelim_3d_box_meta(
    centers_2d: torch.Tensor,
    depths: torch.Tensor,
    dims_3d: torch.Tensor,
    rots: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((centers_2d.shape[0], 7), device="meta", dtype=centers_2d.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("centerfusion_prelim_3d_box", _centerfusion_prelim_3d_box_meta)


def centerfusion_prelim_3d_box(
    centers_2d: torch.Tensor,
    depths: torch.Tensor,
    dims_3d: torch.Tensor,
    rots: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.centerfusion_prelim_3d_box(
        centers_2d,
        depths,
        dims_3d,
        rots,
        K,
    )
