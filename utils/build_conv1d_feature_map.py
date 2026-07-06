# -*- coding: utf-8 -*-
"""
build_conv1d_feature_map

说明：
    将序列特征转换为 Conv1d 默认输入格式。
    该算子用于 F-ConvNet frustum 序列特征从 (B, L, C) 或 (L, C)
    转换为 (B, C, L)。

Args:
    sequence_features: (L, C) 或 (B, L, C)
        按序列维排列的特征。

Returns:
    conv1d_features: (1, C, L) 或 (B, C, L)
        Conv1d 可直接消费的特征图。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _build_conv1d_feature_map_impl(sequence_features: torch.Tensor) -> torch.Tensor:
    if sequence_features.dim() == 2:
        sequence_features = sequence_features.unsqueeze(0)
    if sequence_features.dim() != 3:
        raise ValueError("sequence_features must have shape (L,C) or (B,L,C)")
    return sequence_features.permute(0, 2, 1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("build_conv1d_feature_map", _build_conv1d_feature_map_impl)


def _build_conv1d_feature_map_meta(sequence_features: torch.Tensor) -> torch.Tensor:
    if sequence_features.dim() == 2:
        L, C = sequence_features.shape
        return torch.empty((1, C, L), device="meta", dtype=sequence_features.dtype)
    B, L, C = sequence_features.shape
    return torch.empty((B, C, L), device="meta", dtype=sequence_features.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("build_conv1d_feature_map", _build_conv1d_feature_map_meta)


def build_conv1d_feature_map(sequence_features: torch.Tensor) -> torch.Tensor:
    return torch.ops.fusionrepo.build_conv1d_feature_map(sequence_features)
