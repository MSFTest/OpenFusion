# -*- coding: utf-8 -*-
"""
score_threshold_filter

说明：
    该文件实现基于分数阈值的候选框过滤通用函数算子，
    用于在 proposal / detection 后处理中，
    同步过滤 box 张量与对应 score 张量。

功能：
    根据给定阈值 threshold，
    保留 scores >= threshold 的候选框及其分数。

输入：
    boxes : Tensor
        候选框张量，形状通常为 (N, D)

    scores : Tensor
        候选框对应分数，形状为 (N,)

    threshold : float
        分数阈值

输出：
    filtered_boxes : Tensor
        过滤后的候选框

    filtered_scores : Tensor
        过滤后的分数
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _score_threshold_filter_impl(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float):
    if boxes.dim() != 2:
        raise ValueError("boxes must have shape (N, D)")

    if scores.dim() != 1:
        raise ValueError("scores must have shape (N,)")

    if boxes.size(0) != scores.size(0):
        raise ValueError("boxes and scores must have the same first dimension")

    mask = scores >= float(threshold)

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    return filtered_boxes, filtered_scores


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("score_threshold_filter", _score_threshold_filter_impl)


# ============================================================
# Meta
# ============================================================

def _score_threshold_filter_meta(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float):
    d = boxes.shape[1]

    filtered_boxes = torch.empty(
        (0, d),
        device="meta",
        dtype=boxes.dtype
    )

    filtered_scores = torch.empty(
        (0,),
        device="meta",
        dtype=scores.dtype
    )

    return filtered_boxes, filtered_scores


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("score_threshold_filter", _score_threshold_filter_meta)


# ============================================================
# Wrapper
# ============================================================

def score_threshold_filter(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float):
    return torch.ops.fusionrepo.score_threshold_filter(
        boxes,
        scores,
        float(threshold)
    )