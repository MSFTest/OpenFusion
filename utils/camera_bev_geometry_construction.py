# -*- coding: utf-8 -*-
"""
camera_bev_geometry_construction

说明：
    3D-CVF 相机 BEV 几何构造适配算子。

    该算子保留 3D-CVF 原始输入输出语义：将 BEV 网格中心点通过外参矩阵
    变换到相机坐标系，并归一化为相机坐标系下的方向向量。

Args:
    bev_coords:
        BEV 网格中心点三维坐标，shape 为 (M, 3)，格式为 (x, y, z)。

    intrinsics:
        相机内参矩阵，shape 为 (B, 3, 3)。该参数为兼容 3D-CVF 原接口保留，
        当前几何构造逻辑不使用内参。

    extrinsics:
        外参矩阵，shape 为 (B, 4, 4)，表示输入 BEV 坐标到相机坐标系的变换。

Returns:
    camera_rays:
        相机坐标系下归一化方向向量，shape 为 (B, M, 3)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _camera_bev_geometry_construction_impl(
    bev_coords: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    if bev_coords.dim() != 2 or bev_coords.shape[1] != 3:
        raise ValueError("bev_coords must have shape (M, 3)")
    if intrinsics.dim() != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape (B, 3, 3)")
    if extrinsics.dim() != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError("extrinsics must have shape (B, 4, 4)")
    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError("intrinsics and extrinsics must have the same batch size")

    return torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        bev_coords,
        extrinsics,
        True,
        1e-6,
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "camera_bev_geometry_construction",
    _camera_bev_geometry_construction_impl,
)


def _camera_bev_geometry_construction_meta(
    bev_coords: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (extrinsics.shape[0], bev_coords.shape[0], 3),
        device="meta",
        dtype=bev_coords.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "camera_bev_geometry_construction",
    _camera_bev_geometry_construction_meta,
)


def camera_bev_geometry_construction(
    bev_coords: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.camera_bev_geometry_construction(
        bev_coords,
        intrinsics,
        extrinsics,
    )
