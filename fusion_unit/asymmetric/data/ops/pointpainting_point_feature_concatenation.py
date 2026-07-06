# -*- coding: utf-8 -*-
"""
pointpainting_point_feature_concatenation

说明：
    PointPainting 私有补全算子。复用 fusionrepo 通用特征拼接算子，
    将 LiDAR 点云原始特征与对应点级语义得分拼接，形成 painted points。

Args:
    points_lidar:
        LiDAR 点云原始特征，shape 为 (N, D)。

    point_semantic_scores:
        点级语义类别得分，shape 为 (N, C)。

Returns:
    painted_points:
        融合后的点云特征，shape 为 (N, D + C)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _pointpainting_point_feature_concatenation_impl(
    points_lidar: torch.Tensor,
    point_semantic_scores: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_concatenation(
        [points_lidar, point_semantic_scores],
        1,
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pointpainting_point_feature_concatenation", _pointpainting_point_feature_concatenation_impl)


def _pointpainting_point_feature_concatenation_meta(
    points_lidar: torch.Tensor,
    point_semantic_scores: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (points_lidar.shape[0], points_lidar.shape[1] + point_semantic_scores.shape[1]),
        device="meta",
        dtype=points_lidar.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pointpainting_point_feature_concatenation", _pointpainting_point_feature_concatenation_meta)


def pointpainting_point_feature_concatenation(
    points_lidar: torch.Tensor,
    point_semantic_scores: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pointpainting_point_feature_concatenation(
        points_lidar,
        point_semantic_scores,
    )
