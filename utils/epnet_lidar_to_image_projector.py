# -*- coding: utf-8 -*-
"""
epnet_lidar_to_image_projector

说明：
    EPNeT 私有投影补全算子，注册为：
        torch.ops.fusionrepo.epnet_lidar_to_image_projector(lidar_points, R, t, K)

    该算子保持 EPNeT 原始三返回值语义：将 LiDAR 坐标系三维点投影到
    图像像素坐标系，返回像素坐标 uv、投影深度 depth 和完整投影矩阵 M。
    坐标系约定为 M = K [R|t]，其中 [R|t] 表示 LiDAR 到相机坐标系外参。

Args:
    lidar_points: Tensor
        LiDAR 三维点，shape 为 (B, N, 3)。
    R: Tensor
        外参旋转矩阵，shape 为 (B, 3, 3)。
    t: Tensor
        外参平移向量，shape 为 (B, 3, 1)。
    K: Tensor
        相机内参矩阵，shape 为 (B, 3, 3)。

Returns:
    Tensor[]
        uv: (B, N, 2)，图像像素坐标。
        depth: (B, N, 1)，齐次投影第三通道深度。
        M: (B, 3, 4)，完整投影矩阵 K [R|t]。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_lidar_to_image_projector_impl(
    lidar_points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    K: torch.Tensor,
) -> list[torch.Tensor]:
    if lidar_points.dim() != 3 or lidar_points.shape[-1] != 3:
        raise ValueError("lidar_points must have shape (B, N, 3)")
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must have shape (B, 3, 3)")
    if t.shape[-2:] != (3, 1):
        raise ValueError("t must have shape (B, 3, 1)")
    if K.shape[-2:] != (3, 3):
        raise ValueError("K must have shape (B, 3, 3)")

    Rt = torch.cat([R, t], dim=-1)
    M = torch.ops.fusionrepo.build_projection_matrix(K, Rt)
    uv = torch.ops.fusionrepo.points_to_image(lidar_points, M, 1e-6)

    ones = torch.ones(
        (*lidar_points.shape[:-1], 1),
        device=lidar_points.device,
        dtype=lidar_points.dtype,
    )
    lidar_homo = torch.cat([lidar_points, ones], dim=-1)
    uv_depth = lidar_homo @ M.transpose(-1, -2)
    depth = uv_depth[..., 2:3]

    return [uv.contiguous(), depth.contiguous(), M.contiguous()]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_lidar_to_image_projector", _epnet_lidar_to_image_projector_impl)


def _epnet_lidar_to_image_projector_meta(
    lidar_points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    K: torch.Tensor,
) -> list[torch.Tensor]:
    B, N, _ = lidar_points.shape
    return [
        torch.empty((B, N, 2), device="meta", dtype=lidar_points.dtype),
        torch.empty((B, N, 1), device="meta", dtype=lidar_points.dtype),
        torch.empty((B, 3, 4), device="meta", dtype=lidar_points.dtype),
    ]


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_lidar_to_image_projector", _epnet_lidar_to_image_projector_meta)


def epnet_lidar_to_image_projector(
    lidar_points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    K: torch.Tensor,
) -> list[torch.Tensor]:
    return torch.ops.fusionrepo.epnet_lidar_to_image_projector(
        lidar_points,
        R,
        t,
        K,
    )
