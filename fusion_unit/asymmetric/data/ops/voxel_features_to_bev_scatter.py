# -*- coding: utf-8 -*-
"""
voxel_features_to_bev_scatter

说明：
    将稀疏体素特征按照体素坐标散射到 dense BEV 特征图。

Args:
    voxel_features:
        稀疏体素特征，shape 为 (N, C)。

    voxel_coords:
        体素坐标，shape 为 (N, 4)，格式为 (batch_idx, z, y, x)。
        本算子按 batch_idx、y、x 写入 BEV 平面，z 维不参与输出索引。

    batch_size:
        输出 batch 大小 B。

    bev_h:
        BEV 特征图高度 H。

    bev_w:
        BEV 特征图宽度 W。

    reduce:
        同一 BEV 网格出现多个体素时的归约方式，支持 "assign"、"sum"、"mean"。
        "assign" 保留后写入值，匹配 3D-CVF 原始行为。

Returns:
    bev_features:
        dense BEV 特征图，shape 为 (B, C, H, W)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _voxel_features_to_bev_scatter_impl(
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    batch_size: int,
    bev_h: int,
    bev_w: int,
    reduce: str = "assign",
) -> torch.Tensor:
    if voxel_features.dim() != 2:
        raise ValueError("voxel_features must have shape (N, C)")
    if voxel_coords.dim() != 2 or voxel_coords.shape[1] != 4:
        raise ValueError("voxel_coords must have shape (N, 4)")
    if voxel_features.shape[0] != voxel_coords.shape[0]:
        raise ValueError("voxel_features and voxel_coords must have the same N")
    if batch_size <= 0 or bev_h <= 0 or bev_w <= 0:
        raise ValueError("batch_size, bev_h and bev_w must be positive")

    n, channels = voxel_features.shape
    bev = voxel_features.new_zeros((batch_size, channels, bev_h, bev_w))
    if n == 0:
        return bev

    coords = voxel_coords.to(device=voxel_features.device, dtype=torch.long)
    b = coords[:, 0]
    y = coords[:, 2]
    x = coords[:, 3]
    valid = (b >= 0) & (b < batch_size) & (y >= 0) & (y < bev_h) & (x >= 0) & (x < bev_w)

    if not torch.any(valid):
        return bev

    b = b[valid]
    y = y[valid]
    x = x[valid]
    feat = voxel_features[valid]

    if reduce == "assign":
        bev[b, :, y, x] = feat
        return bev

    linear = b * (bev_h * bev_w) + y * bev_w + x
    bev_flat = bev.permute(0, 2, 3, 1).reshape(batch_size * bev_h * bev_w, channels)
    bev_flat.index_add_(0, linear, feat)

    if reduce == "sum":
        return bev_flat.reshape(batch_size, bev_h, bev_w, channels).permute(0, 3, 1, 2).contiguous()

    if reduce == "mean":
        counts = voxel_features.new_zeros((batch_size * bev_h * bev_w, 1))
        ones = voxel_features.new_ones((feat.shape[0], 1))
        counts.index_add_(0, linear, ones)
        bev_flat = bev_flat / counts.clamp_min(1.0)
        return bev_flat.reshape(batch_size, bev_h, bev_w, channels).permute(0, 3, 1, 2).contiguous()

    raise ValueError("reduce must be one of 'assign', 'sum', 'mean'")


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "voxel_features_to_bev_scatter",
    _voxel_features_to_bev_scatter_impl,
)


def _voxel_features_to_bev_scatter_meta(
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    batch_size: int,
    bev_h: int,
    bev_w: int,
    reduce: str = "assign",
) -> torch.Tensor:
    return torch.empty(
        (int(batch_size), voxel_features.shape[1], int(bev_h), int(bev_w)),
        device="meta",
        dtype=voxel_features.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "voxel_features_to_bev_scatter",
    _voxel_features_to_bev_scatter_meta,
)


def voxel_features_to_bev_scatter(
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    batch_size: int,
    bev_h: int,
    bev_w: int,
    reduce: str = "assign",
) -> torch.Tensor:
    return torch.ops.fusionrepo.voxel_features_to_bev_scatter(
        voxel_features,
        voxel_coords,
        int(batch_size),
        int(bev_h),
        int(bev_w),
        str(reduce),
    )
