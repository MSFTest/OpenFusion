# -*- coding: utf-8 -*-
"""BEV 回归中心到增强 ego 参考点的解码算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _build_reference_points_impl(
    center: torch.Tensor,
    height: torch.Tensor,
    voxel_size: torch.Tensor,
    pc_range: torch.Tensor,
    out_size_factor: int,
) -> torch.Tensor:
    """
    将检测 head 的 BEV center/height 解码为 ego 三维参考点。

    Args:
        center: BEV feature 网格中心，shape 为 ``(B,2,K)``。
        height: ego 高度，shape 为 ``(B,1,K)``。
        voxel_size: BEV voxel 尺寸 ``(vx,vy)``。
        pc_range: BEV 起点 ``(x_min,y_min)``。
        out_size_factor: 检测 head 相对 voxel grid 的下采样倍数。

    Returns:
        Tensor: 增强 ego 坐标参考点，shape 为 ``(B,K,3)``。
    """
    center_x = center[:, 0:1] * float(out_size_factor) * voxel_size[0].to(center)
    center_y = center[:, 1:2] * float(out_size_factor) * voxel_size[1].to(center)
    center_x = center_x + pc_range[0].to(center)
    center_y = center_y + pc_range[1].to(center)
    return torch.cat([center_x, center_y, height], dim=1).permute(0, 2, 1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("build_reference_points", _build_reference_points_impl)


def _build_reference_points_meta(center, height, voxel_size, pc_range, out_size_factor):
    return torch.empty((center.shape[0], center.shape[2], 3), device="meta", dtype=center.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("build_reference_points", _build_reference_points_meta)


def build_reference_points(center, height, voxel_size, pc_range, out_size_factor):
    """调用 ``fusionrepo::build_reference_points``。"""
    return torch.ops.fusionrepo.build_reference_points(
        center, height, voxel_size, pc_range, int(out_size_factor)
    )
