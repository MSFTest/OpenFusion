# -*- coding: utf-8 -*-
"""
centerfusion_radar_pillar_expand

说明：
    CenterFusion 私有雷达几何算子。将雷达点在 z 轴方向扩展为固定
    高度的竖直 pillar，用于后续雷达 pillar 与目标 frustum 的空间关联。

Args:
    radar_points: (N, C)
        雷达点云，前三维为 (x, y, z)。
    pillar_height: float
        每个 pillar 的高度。
    z_center: float
        兼容原始接口的保留参数，不参与计算。

Returns:
    pillars: (N, 2, 3)
        每个 pillar 的 bottom/top 两个端点，坐标系与输入 radar_points 一致。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _centerfusion_radar_pillar_expand_impl(
    radar_points: torch.Tensor,
    pillar_height: float,
    z_center: float = 0.0,
) -> torch.Tensor:
    if radar_points.dim() != 2 or radar_points.shape[1] < 3:
        raise ValueError("radar_points must have shape (N, C>=3)")

    xyz = radar_points[:, :3]
    z_bottom = xyz[:, 2] - float(pillar_height) * 0.5
    z_top = xyz[:, 2] + float(pillar_height) * 0.5

    bottom = torch.stack([xyz[:, 0], xyz[:, 1], z_bottom], dim=1)
    top = torch.stack([xyz[:, 0], xyz[:, 1], z_top], dim=1)

    return torch.stack([bottom, top], dim=1)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("centerfusion_radar_pillar_expand", _centerfusion_radar_pillar_expand_impl)


def _centerfusion_radar_pillar_expand_meta(
    radar_points: torch.Tensor,
    pillar_height: float,
    z_center: float = 0.0,
) -> torch.Tensor:
    return torch.empty(
        (radar_points.shape[0], 2, 3),
        device="meta",
        dtype=radar_points.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("centerfusion_radar_pillar_expand", _centerfusion_radar_pillar_expand_meta)


def centerfusion_radar_pillar_expand(
    radar_points: torch.Tensor,
    pillar_height: float,
    z_center: float = 0.0,
) -> torch.Tensor:
    return torch.ops.fusionrepo.centerfusion_radar_pillar_expand(
        radar_points,
        float(pillar_height),
        float(z_center),
    )
