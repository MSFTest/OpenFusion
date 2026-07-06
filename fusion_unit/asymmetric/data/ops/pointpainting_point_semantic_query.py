# -*- coding: utf-8 -*-
"""
pointpainting_point_semantic_query

说明：
    PointPainting 私有补全算子。根据点投影到图像平面的像素坐标，
    从语义分割得分图中读取每个点对应的语义类别得分。

Args:
    pixel_coords:
        点云投影后的像素坐标，shape 为 (N, 2)，格式为 (u, v)。

    semantic_score_map:
        图像语义分割得分图，shape 为 (H, W, C)。

Returns:
    point_semantic_scores:
        每个点对应的语义类别得分，shape 为 (N, C)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _pointpainting_point_semantic_query_impl(
    pixel_coords: torch.Tensor,
    semantic_score_map: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_gather_by_pixel_index(
        pixel_coords,
        semantic_score_map,
        "long",
        "HWC",
        True,
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pointpainting_point_semantic_query", _pointpainting_point_semantic_query_impl)


def _pointpainting_point_semantic_query_meta(
    pixel_coords: torch.Tensor,
    semantic_score_map: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (pixel_coords.shape[0], semantic_score_map.shape[-1]),
        device="meta",
        dtype=semantic_score_map.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pointpainting_point_semantic_query", _pointpainting_point_semantic_query_meta)


def pointpainting_point_semantic_query(
    pixel_coords: torch.Tensor,
    semantic_score_map: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pointpainting_point_semantic_query(pixel_coords, semantic_score_map)
