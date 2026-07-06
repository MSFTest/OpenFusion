# -*- coding: utf-8 -*-
"""
feature_gate_multiply

说明：
    通用特征门控乘法融合算子，注册为：
        torch.ops.fusionrepo.feature_gate_multiply(x, gate, apply_sigmoid)

    该算子用于多模态或多源特征已经完成空间/索引对齐后，将主特征与
    gate / attention / weight 特征逐元素相乘，得到门控后的融合特征。

Args:
    x: Tensor
        主输入特征，支持任意可广播形状，例如 (B, C, N) 或 (B, C, H, W)。
    gate: Tensor
        门控权重特征，形状需可广播到 x，例如 (B, 1, N) 或 (B, 1, H, W)。
    apply_sigmoid: bool
        是否先对 gate 执行 sigmoid，将权重压缩到 [0, 1]。

Returns:
    Tensor
        门控后的特征，形状为 x 与 gate 的广播结果。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _feature_gate_multiply_impl(
    x: torch.Tensor,
    gate: torch.Tensor,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    if apply_sigmoid:
        gate = torch.sigmoid(gate)
    return (x * gate).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("feature_gate_multiply", _feature_gate_multiply_impl)


def _feature_gate_multiply_meta(
    x: torch.Tensor,
    gate: torch.Tensor,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    out_shape = torch.broadcast_shapes(tuple(x.shape), tuple(gate.shape))
    return torch.empty(out_shape, device="meta", dtype=x.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("feature_gate_multiply", _feature_gate_multiply_meta)


def feature_gate_multiply(
    x: torch.Tensor,
    gate: torch.Tensor,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_gate_multiply(
        x,
        gate,
        bool(apply_sigmoid),
    )
