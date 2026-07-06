# -*- coding: utf-8 -*-
"""
epnet_image_sampler

说明：
    EPNeT 私有图像特征采样适配算子，注册为：
        torch.ops.fusionrepo.epnet_image_sampler(feature_map, grid)

    该算子保持 EPNeT 原始输入语义，接收形状为 (B, 1, N, 2) 的采样网格，
    并复用通用 grid_sample 特征采样算子输出逐点图像特征。

Args:
    feature_map: Tensor
        图像特征图，shape 为 (B, C, H, W)。
    grid: Tensor
        归一化采样网格，shape 为 (B, 1, N, 2) 或 (B, N, 2)。

Returns:
    Tensor
        逐点图像特征，shape 为 (B, C, N)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_image_sampler_impl(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    sampling_grid = grid.squeeze(1) if grid.dim() == 4 and grid.shape[1] == 1 else grid
    return torch.ops.fusionrepo.feature_sampling_by_grid(
        feature_map,
        sampling_grid,
    ).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_image_sampler", _epnet_image_sampler_impl)


def _epnet_image_sampler_meta(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    B, C, _, _ = feature_map.shape
    N = grid.shape[2] if grid.dim() == 4 and grid.shape[1] == 1 else grid.shape[1]
    return torch.empty((B, C, N), device="meta", dtype=feature_map.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_image_sampler", _epnet_image_sampler_meta)


def epnet_image_sampler(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.epnet_image_sampler(feature_map, grid)
