# -*- coding: utf-8 -*-
"""BEV flatten index 到 cell-center query 坐标的转换算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _compute_query_position_impl(
    topk_indices: torch.Tensor, bev_height: int, bev_width: int
) -> torch.Tensor:
    """
    将 BEV flatten 索引转换为网格中心 ``(x,y)`` 坐标。

    Args:
        topk_indices: flatten 索引，shape 为 ``(B,K)``。
        bev_height: BEV 网格高度。
        bev_width: BEV 网格宽度。

    Returns:
        Tensor: BEV cell-center 坐标，shape 为 ``(B,K,2)``。
    """
    del bev_height
    ys = torch.div(topk_indices.long(), int(bev_width), rounding_mode="floor").float() + 0.5
    xs = torch.remainder(topk_indices.long(), int(bev_width)).float() + 0.5
    return torch.stack([xs, ys], dim=-1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("compute_query_position", _compute_query_position_impl)


def _compute_query_position_meta(topk_indices, bev_height, bev_width):
    return torch.empty((*topk_indices.shape, 2), device="meta", dtype=torch.float32)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("compute_query_position", _compute_query_position_meta)


def compute_query_position(topk_indices, bev_height, bev_width):
    """调用 ``fusionrepo::compute_query_position``。"""
    return torch.ops.fusionrepo.compute_query_position(
        topk_indices, int(bev_height), int(bev_width)
    )
