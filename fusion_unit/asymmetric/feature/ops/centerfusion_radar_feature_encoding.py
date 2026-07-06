# -*- coding: utf-8 -*-
"""
centerfusion_radar_feature_encoding

说明：
    CenterFusion 私有雷达特征编码算子。根据雷达点与目标的关联矩阵，
    将点级雷达测量聚合为目标级雷达物理特征。

Args:
    radar_points: (M, C)
        雷达点云，默认前四维为 (x, y, z, velocity)。
    association: (M, N)
        雷达点或 pillar 到目标的关联矩阵，非零表示关联。

Returns:
    encoded: (N, 3)
        目标级雷达特征，格式为 (depth, velocity_x, velocity_y)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _centerfusion_radar_feature_encoding_impl(
    radar_points: torch.Tensor,
    association: torch.Tensor,
) -> torch.Tensor:
    if radar_points.dim() != 2 or radar_points.shape[1] < 4:
        raise ValueError("radar_points must have shape (M, C>=4)")
    if association.dim() != 2 or association.shape[0] != radar_points.shape[0]:
        raise ValueError("association must have shape (M, N)")

    _, num_targets = association.shape
    encoded = torch.zeros((num_targets, 3), device=radar_points.device, dtype=radar_points.dtype)

    for n in range(num_targets):
        mask = association[:, n].bool()
        if mask.any():
            pts = radar_points[mask]
            depth = torch.norm(pts[:, :3], dim=1).min()
            vx = pts[:, 3].mean()
            vy = pts[:, 3].mean()
            encoded[n] = torch.stack([depth, vx, vy])

    return encoded


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("centerfusion_radar_feature_encoding", _centerfusion_radar_feature_encoding_impl)


def _centerfusion_radar_feature_encoding_meta(
    radar_points: torch.Tensor,
    association: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((association.shape[1], 3), device="meta", dtype=radar_points.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("centerfusion_radar_feature_encoding", _centerfusion_radar_feature_encoding_meta)


def centerfusion_radar_feature_encoding(
    radar_points: torch.Tensor,
    association: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.centerfusion_radar_feature_encoding(
        radar_points,
        association,
    )
