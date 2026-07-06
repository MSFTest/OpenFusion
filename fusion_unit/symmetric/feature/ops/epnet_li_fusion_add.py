# -*- coding: utf-8 -*-
"""
epnet_li_fusion_add

说明：
    EPNeT 私有加法融合补全算子，注册为：
        torch.ops.fusionrepo.epnet_li_fusion_add(Fp, Fi, apply_tanh)

    保持 EPNeT 原始语义：点云特征与点位图像特征逐元素相加，并可选执行
    tanh 非线性约束。

Args:
    Fp: Tensor
        点云特征，shape 为 (B, D, N)。
    Fi: Tensor
        图像点级特征，shape 为 (B, D, N)。
    apply_tanh: bool
        是否对相加结果执行 tanh。

Returns:
    Tensor
        融合特征，shape 为 (B, D, N)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_li_fusion_add_impl(
    Fp: torch.Tensor,
    Fi: torch.Tensor,
    apply_tanh: bool = True,
) -> torch.Tensor:
    features = torch.stack([Fp, Fi], dim=0)
    out = torch.ops.fusionrepo.feature_fusion_broadcast_weighted_sum(features, None)
    if apply_tanh:
        out = torch.tanh(out)
    return out.contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_li_fusion_add", _epnet_li_fusion_add_impl)


def _epnet_li_fusion_add_meta(
    Fp: torch.Tensor,
    Fi: torch.Tensor,
    apply_tanh: bool = True,
) -> torch.Tensor:
    return torch.empty_like(Fp, device="meta")


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_li_fusion_add", _epnet_li_fusion_add_meta)


def epnet_li_fusion_add(
    Fp: torch.Tensor,
    Fi: torch.Tensor,
    apply_tanh: bool = True,
) -> torch.Tensor:
    return torch.ops.fusionrepo.epnet_li_fusion_add(Fp, Fi, bool(apply_tanh))
