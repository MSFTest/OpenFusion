# -*- coding: utf-8 -*-
"""
epnet_li_fusion_concat

说明：
    EPNeT 私有拼接融合适配算子，注册为：
        torch.ops.fusionrepo.epnet_li_fusion_concat(tensors, dim)

    保留 EPNeT 原始特征拼接语义，并复用通用特征拼接融合算子。

Args:
    tensors: Tensor[]
        待拼接的特征张量列表，例如两个 (B, D, N) 张量。
    dim: int
        拼接维度，默认通道维 1。

Returns:
    Tensor
        拼接后的融合特征。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_li_fusion_concat_impl(
    tensors: list[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_concatenation(
        tensors,
        int(dim),
    ).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_li_fusion_concat", _epnet_li_fusion_concat_impl)


def _epnet_li_fusion_concat_meta(
    tensors: list[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    out_shape = list(tensors[0].shape)
    out_shape[int(dim)] = sum(t.shape[int(dim)] for t in tensors)
    return torch.empty(tuple(out_shape), device="meta", dtype=tensors[0].dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_li_fusion_concat", _epnet_li_fusion_concat_meta)


def epnet_li_fusion_concat(
    tensors: list[torch.Tensor],
    dim: int = 1,
) -> torch.Tensor:
    return torch.ops.fusionrepo.epnet_li_fusion_concat(tensors, int(dim))
