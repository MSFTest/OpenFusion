# -*- coding: utf-8 -*-
"""
epnet_li_fusion_muti

说明：
    EPNeT 私有乘法门控融合适配算子，注册为：
        torch.ops.fusionrepo.epnet_li_fusion_muti(x, y, sigmoid_y)

    该算子保留原始 EPNeT 拼写和输入输出语义，并复用通用门控乘法算子。

Args:
    x: Tensor
        主输入特征，shape 通常为 (B, D, N)。
    y: Tensor
        权重或门控特征，shape 可为 (B, D, N) 或 (B, 1, N)。
    sigmoid_y: bool
        是否先对 y 执行 sigmoid。

Returns:
    Tensor
        加权后的融合特征。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_li_fusion_muti_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmoid_y: bool = False,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_gate_multiply(
        x,
        y,
        bool(sigmoid_y),
    ).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_li_fusion_muti", _epnet_li_fusion_muti_impl)


def _epnet_li_fusion_muti_meta(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmoid_y: bool = False,
) -> torch.Tensor:
    out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
    return torch.empty(out_shape, device="meta", dtype=x.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_li_fusion_muti", _epnet_li_fusion_muti_meta)


def epnet_li_fusion_muti(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmoid_y: bool = False,
) -> torch.Tensor:
    return torch.ops.fusionrepo.epnet_li_fusion_muti(x, y, bool(sigmoid_y))
