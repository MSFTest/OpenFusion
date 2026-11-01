# -*- coding: utf-8 -*-
"""BEVDet dense heatmap 跨类别 Top-K proposal 选择算子。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library


def _select_topk_proposals_impl(
    heatmap: torch.Tensor, num_proposals: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从所有类别和 BEV 网格中选择最高分候选。

    Args:
        heatmap: 筛选后 heatmap，shape 为 ``(B,C,H,W)``。
        num_proposals: 每个 batch 的候选数量 K。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: 类别 ``(B,K)``、BEV flatten
        索引 ``(B,K)``、分数 ``(B,K)`` 和 ``(x,y)`` 坐标 ``(B,K,2)``。
    """
    bsz, _, height, width = heatmap.shape
    flat = heatmap.reshape(bsz, -1)
    k = min(int(num_proposals), flat.shape[1])
    scores, proposals = torch.topk(flat, k=k, dim=-1)
    spatial_size = height * width
    classes = torch.div(proposals, spatial_size, rounding_mode="floor").long()
    indices = torch.remainder(proposals, spatial_size).long()
    ys = torch.div(indices, width, rounding_mode="floor")
    xs = torch.remainder(indices, width)
    return classes, indices, scores, torch.stack([xs, ys], dim=-1).long()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("select_topk_proposals", _select_topk_proposals_impl)


def _select_topk_proposals_meta(heatmap, num_proposals):
    bsz = heatmap.shape[0]
    k = min(int(num_proposals), heatmap.shape[1] * heatmap.shape[2] * heatmap.shape[3])
    return (
        torch.empty((bsz, k), device="meta", dtype=torch.long),
        torch.empty((bsz, k), device="meta", dtype=torch.long),
        torch.empty((bsz, k), device="meta", dtype=heatmap.dtype),
        torch.empty((bsz, k, 2), device="meta", dtype=torch.long),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("select_topk_proposals", _select_topk_proposals_meta)


def select_topk_proposals(heatmap, num_proposals):
    """调用 ``fusionrepo::select_topk_proposals``。"""
    return torch.ops.fusionrepo.select_topk_proposals(heatmap, int(num_proposals))
