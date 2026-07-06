# -*- coding: utf-8 -*-
"""
masked_reduce_pool

说明：
    带有效掩码的分组特征归约池化算子。
    该算子用于 frustum、voxel、ROI 等分组特征中屏蔽 padding 点后，
    按指定维度执行 max、mean 或 sum 聚合。

Args:
    x: Tensor
        输入分组特征，例如 (B, G, M, C)、(G, M, C) 或其他形状。

    mask: Tensor
        有效掩码，形状应等于 x 去掉特征维后的形状，或可广播到 x。
        True 表示有效元素，False 表示 padding。

    dim: int
        归约维度。

    mode: str
        归约方式，支持 "max"、"mean"、"sum"。

    keepdim: bool
        是否保留归约维度。

Returns:
    pooled: Tensor
        归约后的特征张量。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _masked_reduce_pool_impl(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    mode: str = "max",
    keepdim: bool = False,
) -> torch.Tensor:
    dim = int(dim)
    if dim < 0:
        dim += x.dim()
    if dim < 0 or dim >= x.dim():
        raise ValueError("dim out of range")

    mask_bool = mask.to(device=x.device, dtype=torch.bool)
    while mask_bool.dim() < x.dim():
        mask_bool = mask_bool.unsqueeze(-1)
    mask_bool = mask_bool.expand_as(x)

    if mode == "max":
        if x.dtype.is_floating_point:
            fill_value = torch.finfo(x.dtype).min
        else:
            fill_value = torch.iinfo(x.dtype).min
        masked = x.masked_fill(~mask_bool, fill_value)
        out = masked.amax(dim=dim, keepdim=keepdim)
        if keepdim:
            valid_any = mask_bool.any(dim=dim, keepdim=True)
        else:
            valid_any = mask_bool.any(dim=dim, keepdim=False)
        return torch.where(valid_any, out, torch.zeros_like(out))

    x_masked = x * mask_bool.to(dtype=x.dtype)
    if mode == "sum":
        return x_masked.sum(dim=dim, keepdim=keepdim)

    if mode == "mean":
        summed = x_masked.sum(dim=dim, keepdim=keepdim)
        counts = mask_bool.to(dtype=x.dtype).sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
        return summed / counts

    raise ValueError(f"Unsupported reduce mode: {mode}")


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("masked_reduce_pool", _masked_reduce_pool_impl)


def _masked_reduce_pool_meta(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    mode: str = "max",
    keepdim: bool = False,
) -> torch.Tensor:
    dim = int(dim)
    if dim < 0:
        dim += x.dim()
    shape = list(x.shape)
    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)
    return torch.empty(tuple(shape), device="meta", dtype=x.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("masked_reduce_pool", _masked_reduce_pool_meta)


def masked_reduce_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    mode: str = "max",
    keepdim: bool = False,
) -> torch.Tensor:
    return torch.ops.fusionrepo.masked_reduce_pool(
        x,
        mask,
        int(dim),
        str(mode),
        bool(keepdim),
    )
