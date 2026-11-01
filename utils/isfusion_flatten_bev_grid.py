# -*- coding: utf-8 -*-
"""
IS-Fusion BEV 特征展平与坐标生成算子。

Args:
    bev_feats: BEV 特征图，shape 为 (B, C, H, W)，坐标系为 ego BEV grid。

Returns:
    tuple[Tensor, Tensor]:
        - grid_features: 展平 BEV token，shape 为 (B*H*W, C)。
        - grid_coords: token 坐标，shape 为 (B*H*W, 4)，顺序为 (batch, z, y, x)，z 固定为 0。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library


def _isfusion_flatten_bev_grid_impl(bev_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if bev_feats.dim() != 4:
        raise ValueError("bev_feats must have shape (B,C,H,W)")
    batch_size, channels, height, width = bev_feats.shape
    grid_features = bev_feats.flatten(2).transpose(1, 2).reshape(-1, channels).contiguous()
    y, x = torch.meshgrid(
        torch.arange(height, device=bev_feats.device),
        torch.arange(width, device=bev_feats.device),
        indexing="ij",
    )
    z = torch.zeros_like(y)
    coords = []
    for batch_idx in range(batch_size):
        b = torch.full_like(y, batch_idx)
        coords.append(torch.stack([b, z, y, x], dim=-1).reshape(-1, 4))
    return grid_features, torch.cat(coords, dim=0).to(torch.long)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("isfusion_flatten_bev_grid", _isfusion_flatten_bev_grid_impl)


def _isfusion_flatten_bev_grid_meta(bev_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, channels, height, width = bev_feats.shape
    n = batch_size * height * width
    return (
        torch.empty((n, channels), device="meta", dtype=bev_feats.dtype),
        torch.empty((n, 4), device="meta", dtype=torch.long),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("isfusion_flatten_bev_grid", _isfusion_flatten_bev_grid_meta)


def isfusion_flatten_bev_grid(bev_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.fusionrepo.isfusion_flatten_bev_grid(bev_feats)
