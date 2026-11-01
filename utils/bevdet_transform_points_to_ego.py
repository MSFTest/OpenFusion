# -*- coding: utf-8 -*-
"""BEVDet 点云到目标 ego 坐标系的变换算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _transform_points_to_ego_impl(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    使用齐次矩阵变换点云，并保留 xyz 之外的附加字段。

    Args:
        points: 点云，shape 为 ``(B, P, C)``，前三维是源坐标系 xyz。
        transform: 源坐标系到目标 ego 坐标系的矩阵，shape 为 ``(B, 4, 4)``。

    Returns:
        Tensor: 目标 ego 坐标系点云，shape 为 ``(B, P, C)``。
    """
    transformed_xyz = torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        points[..., :3], transform.to(dtype=points.dtype, device=points.device), False, 1e-6
    )
    if points.shape[-1] == 3:
        return transformed_xyz.contiguous()
    return torch.cat([transformed_xyz, points[..., 3:]], dim=-1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transform_points_to_ego", _transform_points_to_ego_impl)


def _transform_points_to_ego_meta(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return torch.empty(tuple(points.shape), device="meta", dtype=points.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transform_points_to_ego", _transform_points_to_ego_meta)


def transform_points_to_ego(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """调用 ``fusionrepo::transform_points_to_ego``。"""
    return torch.ops.fusionrepo.transform_points_to_ego(points, transform)
