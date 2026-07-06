# -*- coding: utf-8 -*-
"""
centerfusion_radar_frustum_association

说明：
    CenterFusion 私有雷达-目标关联算子。根据雷达 pillar 的中心点与
    frustum 的原始前两个顶点边界，生成 pillar 到目标的关联矩阵。

Args:
    radar_pillars: (M, 2, 3)
        每个雷达 pillar 的 bottom/top 两个端点。
    frustums: (N, 8, 3)
        每个目标 frustum 的 8 个顶点。为保持原始功能，当前实现沿用
        frustums[:, 0, :] 作为最小边界、frustums[:, 1, :] 作为最大边界。

Returns:
    association: (M, N)
        雷达 pillar 与目标 frustum 的 0/1 关联矩阵，dtype 为 int32。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _centerfusion_radar_frustum_association_impl(
    radar_pillars: torch.Tensor,
    frustums: torch.Tensor,
) -> torch.Tensor:
    if radar_pillars.dim() != 3 or radar_pillars.shape[1:] != (2, 3):
        raise ValueError("radar_pillars must have shape (M, 2, 3)")
    if frustums.dim() != 3 or frustums.shape[1:] != (8, 3):
        raise ValueError("frustums must have shape (N, 8, 3)")

    pillar_centers = radar_pillars.mean(dim=1)
    frustum_min = frustums[:, 0, :]
    frustum_max = frustums[:, 1, :]

    pc = pillar_centers.unsqueeze(1)
    inside = (pc >= frustum_min) & (pc <= frustum_max)

    return inside.all(dim=2).to(torch.int32)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "centerfusion_radar_frustum_association",
    _centerfusion_radar_frustum_association_impl,
)


def _centerfusion_radar_frustum_association_meta(
    radar_pillars: torch.Tensor,
    frustums: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (radar_pillars.shape[0], frustums.shape[0]),
        device="meta",
        dtype=torch.int32,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "centerfusion_radar_frustum_association",
    _centerfusion_radar_frustum_association_meta,
)


def centerfusion_radar_frustum_association(
    radar_pillars: torch.Tensor,
    frustums: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.centerfusion_radar_frustum_association(
        radar_pillars,
        frustums,
    )
