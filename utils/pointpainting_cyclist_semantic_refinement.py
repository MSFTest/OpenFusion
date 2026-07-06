# -*- coding: utf-8 -*-
"""
pointpainting_cyclist_semantic_refinement

说明：
    PointPainting 私有规范化算子。根据 cyclist/rider 语义通道置信度，
    对已拼接语义得分的点云特征做一致性修正。

Args:
    painted_points:
        已完成语义拼接的点云特征，shape 为 (N, D + C)。

    cyclist_index:
        cyclist/rider 语义通道索引，支持负索引。

    threshold:
        高置信度阈值。大于等于该阈值的位置置为 1.0，
        小于该阈值的位置乘以 0.5。

Returns:
    refined_points:
        语义修正后的点云特征，shape 与 painted_points 相同。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _pointpainting_cyclist_semantic_refinement_impl(
    painted_points: torch.Tensor,
    cyclist_index: int = -1,
    threshold: float = 0.3,
) -> torch.Tensor:
    if painted_points.dim() != 2:
        raise ValueError("painted_points must have shape (N, C)")

    refined_points = painted_points.clone()
    semantic_scores = refined_points[:, int(cyclist_index)]
    cyclist_mask = semantic_scores >= float(threshold)
    refined_points[cyclist_mask, int(cyclist_index)] = 1.0
    refined_points[~cyclist_mask, int(cyclist_index)] *= 0.5
    return refined_points


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pointpainting_cyclist_semantic_refinement", _pointpainting_cyclist_semantic_refinement_impl)


def _pointpainting_cyclist_semantic_refinement_meta(
    painted_points: torch.Tensor,
    cyclist_index: int = -1,
    threshold: float = 0.3,
) -> torch.Tensor:
    return torch.empty(painted_points.shape, device="meta", dtype=painted_points.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pointpainting_cyclist_semantic_refinement", _pointpainting_cyclist_semantic_refinement_meta)


def pointpainting_cyclist_semantic_refinement(
    painted_points: torch.Tensor,
    cyclist_index: int = -1,
    threshold: float = 0.3,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pointpainting_cyclist_semantic_refinement(
        painted_points,
        int(cyclist_index),
        float(threshold),
    )
