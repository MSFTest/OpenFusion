# -*- coding: utf-8 -*-
"""
feature_concatenation

说明：
    该文件实现通用特征拼接原语，
    供 operator_registry.py 中注册后的

        torch.ops.fusionrepo.feature_concatenation

    调用。

功能：
    将多个特征张量按照指定维度进行拼接。

    本算子只负责：
        torch.cat(features, dim)

    不负责：
        - 线性映射
        - 激活函数
        - 归一化
        - 注意力机制
        - 特征压缩

输入：

    features : Tensor[]
        需要拼接的特征列表

    dim : int
        拼接维度

输出：

    out : Tensor
        拼接后的特征
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _feature_concatenation_impl(
        features: list[torch.Tensor],
        dim: int):

    if len(features) == 0:
        raise ValueError("features list cannot be empty")

    out = torch.cat(features, dim=dim)

    return out


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "feature_concatenation",
    _feature_concatenation_impl
)


# ============================================================
# Meta
# ============================================================

def _feature_concatenation_meta(
        features: list[torch.Tensor],
        dim: int):

    ref = features[0]

    shape = list(ref.shape)

    concat_size = 0
    for t in features:
        concat_size += t.shape[dim]

    shape[dim] = concat_size

    return torch.empty(
        tuple(shape),
        device="meta",
        dtype=ref.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "feature_concatenation",
    _feature_concatenation_meta
)


# ============================================================
# wrapper
# ============================================================

def feature_concatenation(
        features: list[torch.Tensor],
        dim: int):

    return torch.ops.fusionrepo.feature_concatenation(
        features,
        int(dim)
    )