# -*- coding: utf-8 -*-
"""
pixels_depth_to_camera_points

说明：
    将图像像素坐标与深度值按针孔相机模型反投影到相机坐标系。

Args:
    pixels_2d: (..., 2)
        像素坐标，最后一维为 (u, v)。
    depths: (...) 或 (..., 1)
        与 pixels_2d 前缀维度可广播的深度值。
    K: (3, 3)
        相机内参矩阵，坐标系为图像像素坐标系到相机坐标系反投影。

Returns:
    points_cam: (..., 3)
        相机坐标系下的三维点，最后一维为 (x, y, z)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _pixels_depth_to_camera_points_impl(
    pixels_2d: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    if pixels_2d.shape[-1] != 2:
        raise ValueError("pixels_2d last dim must be 2")
    if K.shape[-2:] != (3, 3):
        raise ValueError("K must have shape (3, 3)")

    depth = depths.to(device=pixels_2d.device, dtype=pixels_2d.dtype)
    if depth.dim() > 0 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    u = pixels_2d[..., 0]
    v = pixels_2d[..., 1]

    K = K.to(device=pixels_2d.device, dtype=pixels_2d.dtype)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth.expand_as(x)

    return torch.stack([x, y, z], dim=-1)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pixels_depth_to_camera_points", _pixels_depth_to_camera_points_impl)


def _pixels_depth_to_camera_points_meta(
    pixels_2d: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        pixels_2d.shape[:-1] + (3,),
        device="meta",
        dtype=pixels_2d.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pixels_depth_to_camera_points", _pixels_depth_to_camera_points_meta)


def pixels_depth_to_camera_points(
    pixels_2d: torch.Tensor,
    depths: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pixels_depth_to_camera_points(pixels_2d, depths, K)
