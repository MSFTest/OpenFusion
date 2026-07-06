# -*- coding: utf-8 -*-
"""
epnet_grid_generator

说明：
    EPNeT 私有采样网格补全算子，注册为：
        torch.ops.fusionrepo.epnet_grid_generator(img_w, img_h, uv, depth)

    基于图像像素坐标和投影深度生成 EPNeT 原始格式的归一化坐标、采样网格
    与有效投影掩码。像素坐标系中 uv[..., 0] 为 u，uv[..., 1] 为 v。

Args:
    img_w: int
        图像宽度 W。
    img_h: int
        图像高度 H。
    uv: Tensor
        图像像素坐标，shape 为 (B, N, 2)。
    depth: Tensor
        投影深度，shape 为 (B, N, 1)。

Returns:
    Tensor[]
        xy_norm: (B, N, 2)，grid_sample 归一化坐标。
        grid: (B, 1, N, 2)，EPNeT 原始采样网格格式。
        mask: (B, N)，有效投影 bool 掩码。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_grid_generator_impl(
    img_w: int,
    img_h: int,
    uv: torch.Tensor,
    depth: torch.Tensor,
) -> list[torch.Tensor]:
    xy_norm = torch.ops.fusionrepo.pixel_to_grid_normalized(uv, int(img_h), int(img_w))
    mask = torch.ops.fusionrepo.projected_pixel_valid_mask(
        uv,
        depth,
        int(img_h),
        int(img_w),
        0.0,
    )
    grid = xy_norm.unsqueeze(1)
    return [xy_norm.contiguous(), grid.contiguous(), mask.contiguous()]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_grid_generator", _epnet_grid_generator_impl)


def _epnet_grid_generator_meta(
    img_w: int,
    img_h: int,
    uv: torch.Tensor,
    depth: torch.Tensor,
) -> list[torch.Tensor]:
    B, N, _ = uv.shape
    return [
        torch.empty((B, N, 2), device="meta", dtype=uv.dtype),
        torch.empty((B, 1, N, 2), device="meta", dtype=uv.dtype),
        torch.empty((B, N), device="meta", dtype=torch.bool),
    ]


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_grid_generator", _epnet_grid_generator_meta)


def epnet_grid_generator(
    img_w: int,
    img_h: int,
    uv: torch.Tensor,
    depth: torch.Tensor,
) -> list[torch.Tensor]:
    return torch.ops.fusionrepo.epnet_grid_generator(
        int(img_w),
        int(img_h),
        uv,
        depth,
    )
