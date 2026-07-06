# -*- coding: utf-8 -*-
"""
centerfusion_frustum_from_box3d

说明：
    CenterFusion 私有 frustum 构建算子。根据初步 3D 框中心和尺寸，
    在输入框基础上沿 z 方向加入 frustum_depth 扩展，生成 8 个 frustum 顶点。

Args:
    box3d: (N, 7)
        三维框，格式为 (x, y, z, length, width, height, yaw)。
    K: (3, 3)
        兼容原始接口的相机内参参数，当前原始逻辑不使用该参数。
    frustum_depth: float
        frustum 在 z 方向的额外扩展长度。

Returns:
    corners: (N, 8, 3)
        frustum 的 8 个顶点，坐标系与 box3d 前三维一致。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _centerfusion_frustum_from_box3d_impl(
    box3d: torch.Tensor,
    K: torch.Tensor,
    frustum_depth: float,
) -> torch.Tensor:
    if box3d.dim() != 2 or box3d.shape[1] < 6:
        raise ValueError("box3d must have shape (N, 7)")

    centers = box3d[:, :3]
    dims = box3d[:, 3:6]

    dx = dims[:, 0] * 0.5
    dy = dims[:, 1] * 0.5
    dz = dims[:, 2] * 0.5 + float(frustum_depth) * 0.5

    return torch.stack(
        [
            centers + torch.stack([dx, dy, dz], dim=1),
            centers + torch.stack([dx, -dy, dz], dim=1),
            centers + torch.stack([-dx, -dy, dz], dim=1),
            centers + torch.stack([-dx, dy, dz], dim=1),
            centers + torch.stack([dx, dy, -dz], dim=1),
            centers + torch.stack([dx, -dy, -dz], dim=1),
            centers + torch.stack([-dx, -dy, -dz], dim=1),
            centers + torch.stack([-dx, dy, -dz], dim=1),
        ],
        dim=1,
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("centerfusion_frustum_from_box3d", _centerfusion_frustum_from_box3d_impl)


def _centerfusion_frustum_from_box3d_meta(
    box3d: torch.Tensor,
    K: torch.Tensor,
    frustum_depth: float,
) -> torch.Tensor:
    return torch.empty((box3d.shape[0], 8, 3), device="meta", dtype=box3d.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("centerfusion_frustum_from_box3d", _centerfusion_frustum_from_box3d_meta)


def centerfusion_frustum_from_box3d(
    box3d: torch.Tensor,
    K: torch.Tensor,
    frustum_depth: float,
) -> torch.Tensor:
    return torch.ops.fusionrepo.centerfusion_frustum_from_box3d(
        box3d,
        K,
        float(frustum_depth),
    )
