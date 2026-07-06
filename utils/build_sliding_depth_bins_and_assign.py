# -*- coding: utf-8 -*-
"""
build_sliding_depth_bins_and_assign

说明：
    F-ConvNet sliding frustum 深度分桶适配算子。
    该算子复用 fusionrepo::generate_depth_bins 生成滑动深度区间，
    再复用 fusionrepo::assign_values_to_bins 将每个深度值分配到区间索引。

Args:
    depth: (N,) 或 (B, N)
        点深度值。

    bin_size: float
        单个深度 bin 的长度。

    stride: float
        相邻深度 bin 的滑动步长。

    z_min: float
        深度下界。传入负数时自动使用有效 depth 的最小值。

    z_max: float
        深度上界。传入负数时自动使用有效 depth 的最大值。

Returns:
    bins: (T, 2)
        滑动深度区间定义。

    bin_indices: (N,) 或 (B, N)
        每个深度值对应的 bin 索引，未命中时为 -1。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _build_sliding_depth_bins_and_assign_impl(
    depth: torch.Tensor,
    bin_size: float,
    stride: float,
    z_min: float = -1.0,
    z_max: float = -1.0,
):
    valid_mask = depth > 0
    if float(z_min) < 0:
        z_min_val = depth[valid_mask].amin() if valid_mask.any() else depth.amin()
    else:
        z_min_val = torch.tensor(float(z_min), device=depth.device, dtype=depth.dtype)

    if float(z_max) < 0:
        z_max_val = depth[valid_mask].amax() if valid_mask.any() else depth.amax()
    else:
        z_max_val = torch.tensor(float(z_max), device=depth.device, dtype=depth.dtype)

    depth_range = torch.stack([z_min_val, z_max_val])
    bins = torch.ops.fusionrepo.generate_depth_bins(
        depth_range,
        0,
        2,
        float(bin_size),
        float(stride),
    )
    bin_indices = torch.ops.fusionrepo.assign_values_to_bins(depth, bins, True, -1)
    return bins, bin_indices


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("build_sliding_depth_bins_and_assign", _build_sliding_depth_bins_and_assign_impl)


def _build_sliding_depth_bins_and_assign_meta(
    depth: torch.Tensor,
    bin_size: float,
    stride: float,
    z_min: float = -1.0,
    z_max: float = -1.0,
):
    bins = torch.empty((1, 2), device="meta", dtype=depth.dtype)
    bin_indices = torch.empty(depth.shape, device="meta", dtype=torch.long)
    return bins, bin_indices


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("build_sliding_depth_bins_and_assign", _build_sliding_depth_bins_and_assign_meta)


def build_sliding_depth_bins_and_assign(
    depth: torch.Tensor,
    bin_size: float,
    stride: float,
    z_min: float = -1.0,
    z_max: float = -1.0,
):
    return torch.ops.fusionrepo.build_sliding_depth_bins_and_assign(
        depth,
        float(bin_size),
        float(stride),
        float(z_min),
        float(z_max),
    )
