# -*- coding: utf-8 -*-
"""跨项目 descending-argsort Top-K 索引算子。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library


def _impl(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """保持 SparseFusion/DeepInteraction 的 argsort 排序语义。"""
    if scores.ndim < 2:
        raise ValueError("scores 必须至少包含 batch 维和一个候选维")
    flat = scores.reshape(scores.shape[0], -1)
    if k < 0 or k > flat.shape[1]:
        raise ValueError("k 必须位于 [0, 每批候选总数] 内")
    indices = flat.argsort(-1, descending=True)[:, :k]
    return indices.contiguous(), flat.gather(1, indices).contiguous()


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("descending_argsort_topk", _impl)


def _meta(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim < 2:
        raise ValueError("scores 必须至少包含 batch 维和一个候选维")
    total = 1
    for size in scores.shape[1:]:
        total *= size
    if k < 0 or k > total:
        raise ValueError("k 必须位于 [0, 每批候选总数] 内")
    shape = (scores.shape[0], k)
    return (
        torch.empty(shape, device="meta", dtype=torch.long),
        torch.empty(shape, device="meta", dtype=scores.dtype),
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("descending_argsort_topk", _meta)


def descending_argsort_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 batch 展平候选并以降序 argsort 选前 K 项。

    Args:
        scores: ``[B,...]`` 浮点或整型分数；batch 后各维按 contiguous reshape
            顺序展平，允许 non-contiguous 输入。
        k: 每批返回数量，范围 ``0<=K<=prod(scores.shape[1:])``。

    Returns:
        ``indices long[B,K]`` 和 ``selected_scores[B,K]``。两者与输入同
        device，分数保持输入 dtype；索引/排序离散不可微，gather 后的分数对
        被选输入可微。并列值顺序严格继承当前 PyTorch ``argsort`` 契约。
    """
    return torch.ops.fusionrepo.descending_argsort_topk(scores, int(k))
