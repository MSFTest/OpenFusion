# -*- coding: utf-8 -*-
"""
align_and_concat_multiscale_1d

说明：
    多尺度 1D 特征对齐与通道拼接融合算子。
    该算子以第一个输入特征的序列长度为基准，将后续特征在最后一维裁剪到
    同一长度，然后复用 fusionrepo::feature_concatenation 在通道维执行拼接。

Args:
    features: Tensor[]
        多尺度 1D 特征列表，每个张量形状为 (B, C_i, L_i)。

Returns:
    fused: (B, sum(C_i), L_ref)
        对齐并拼接后的融合特征，L_ref 为第一个特征的序列长度。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _align_and_concat_multiscale_1d_impl(features: list[torch.Tensor]) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("features cannot be empty")
    ref = features[0]
    if ref.dim() != 3:
        raise ValueError("all features must have shape (B, C, L)")

    B, _, L_ref = ref.shape
    aligned = []
    for feat in features:
        if feat.dim() != 3:
            raise ValueError("all features must have shape (B, C, L)")
        if feat.shape[0] != B:
            raise ValueError("all features must share batch size")
        if feat.shape[-1] < L_ref:
            raise ValueError("feature length cannot be shorter than reference length")
        aligned.append(feat[:, :, :L_ref])

    return torch.ops.fusionrepo.feature_concatenation(aligned, 1)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("align_and_concat_multiscale_1d", _align_and_concat_multiscale_1d_impl)


def _align_and_concat_multiscale_1d_meta(features: list[torch.Tensor]) -> torch.Tensor:
    ref = features[0]
    B, _, L_ref = ref.shape
    C_out = sum(feat.shape[1] for feat in features)
    return torch.empty((B, C_out, L_ref), device="meta", dtype=ref.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("align_and_concat_multiscale_1d", _align_and_concat_multiscale_1d_meta)


def align_and_concat_multiscale_1d(features: list[torch.Tensor]) -> torch.Tensor:
    return torch.ops.fusionrepo.align_and_concat_multiscale_1d(features)
