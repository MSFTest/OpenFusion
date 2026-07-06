# -*- coding: utf-8 -*-
"""
assign_values_to_bins

说明：
    将连续标量值分配到区间 bins 中，输出每个值对应的 bin 索引。
    可用于深度、距离、range 等一维标量离散化。

Args:
    values: 任意形状 Tensor
        待分配的连续标量值。

    bins: (T, 2)
        区间定义，每行为 [start, end]。

    include_right: bool
        是否将右边界视为闭区间。False 时使用 [start, end)。

    invalid_index: int
        未落入任何区间时填充的索引值。

Returns:
    bin_indices: 与 values 形状一致的 LongTensor
        每个值所属的 bin 索引。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _assign_values_to_bins_impl(
    values: torch.Tensor,
    bins: torch.Tensor,
    include_right: bool = False,
    invalid_index: int = -1,
) -> torch.Tensor:
    if bins.dim() != 2 or bins.shape[1] != 2:
        raise ValueError("bins must have shape (T, 2)")

    bins = bins.to(device=values.device, dtype=values.dtype)
    flat_values = values.reshape(-1)
    starts = bins[:, 0].view(1, -1)
    ends = bins[:, 1].view(1, -1)
    vals = flat_values.view(-1, 1)

    if include_right:
        inside = (vals >= starts) & (vals <= ends)
    else:
        inside = (vals >= starts) & (vals < ends)

    has_match = inside.any(dim=1)
    bin_ids = torch.arange(bins.shape[0], device=values.device, dtype=torch.long).view(1, -1)
    matched_ids = torch.where(inside, bin_ids, torch.full_like(bin_ids, int(invalid_index)))
    last_match = matched_ids.max(dim=1).values
    invalid = torch.full_like(last_match, int(invalid_index))
    out = torch.where(has_match, last_match, invalid)
    return out.reshape(values.shape)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("assign_values_to_bins", _assign_values_to_bins_impl)


def _assign_values_to_bins_meta(
    values: torch.Tensor,
    bins: torch.Tensor,
    include_right: bool = False,
    invalid_index: int = -1,
) -> torch.Tensor:
    return torch.empty(values.shape, device="meta", dtype=torch.long)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("assign_values_to_bins", _assign_values_to_bins_meta)


def assign_values_to_bins(
    values: torch.Tensor,
    bins: torch.Tensor,
    include_right: bool = False,
    invalid_index: int = -1,
) -> torch.Tensor:
    return torch.ops.fusionrepo.assign_values_to_bins(
        values,
        bins,
        bool(include_right),
        int(invalid_index),
    )
