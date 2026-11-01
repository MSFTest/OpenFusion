# -*- coding: utf-8 -*-
"""
带多视角与掩码支持的通用 grid_sample 特征采样算子。

Args:
    feature_map: 特征图，shape 为 (B, C, H, W) 或 (B, V, C, H, W)。
    grid: 归一化采样坐标，shape 为 (B, N, 2)、(B, N, 1, 2)、
        (B, V, N, 2) 或 (B, V, P, L, 2)，坐标范围为 [-1, 1]。
    valid_mask: 可选有效掩码，shape 为 grid.shape[:-1]。
    align_corners: 传递给 torch.nn.functional.grid_sample 的对齐方式。
    padding_mode: 越界采样填充方式，通常为 "zeros"。

Returns:
    Tensor: 采样特征。输入含视角维且 grid 为 (B,V,P,L,2) 时返回
    (B,V,C,P,L)，否则返回 (B,C,N) 或 (B,V,C,N)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library


def _sample_features_by_grid_v2_impl(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
    valid_mask: torch.Tensor = None,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    if feature_map.dim() == 4:
        batch_size, channels = feature_map.shape[:2]
        if grid.dim() == 3:
            sample_shape = grid.shape[1:2]
            grid_in = grid.unsqueeze(2)
        elif grid.dim() == 4:
            sample_shape = (grid.shape[1],)
            grid_in = grid
        else:
            raise ValueError("grid for 4D feature_map must have shape (B,N,2) or (B,N,1,2)")
        sampled = F.grid_sample(feature_map, grid_in.to(feature_map.dtype), mode="bilinear", padding_mode=padding_mode, align_corners=bool(align_corners))
        out = sampled.squeeze(-1) if sampled.shape[-1] == 1 else sampled.reshape(batch_size, channels, -1)
        if valid_mask is not None:
            out = out * valid_mask.reshape(batch_size, -1).to(out.dtype)[:, None, :]
        return out.contiguous()

    if feature_map.dim() != 5:
        raise ValueError("feature_map must have shape (B,C,H,W) or (B,V,C,H,W)")
    batch_size, num_views, channels = feature_map.shape[:3]
    if grid.dim() == 4:
        sample_shape = (grid.shape[2],)
        grid_in = grid.reshape(batch_size * num_views, grid.shape[2], 1, 2)
    elif grid.dim() == 5:
        sample_shape = grid.shape[2:4]
        grid_in = grid.reshape(batch_size * num_views, sample_shape[0] * sample_shape[1], 1, 2)
    else:
        raise ValueError("grid for 5D feature_map must have shape (B,V,N,2) or (B,V,P,L,2)")
    feat = feature_map.reshape(batch_size * num_views, channels, feature_map.shape[-2], feature_map.shape[-1])
    sampled = F.grid_sample(feat, grid_in.to(feat.dtype), mode="bilinear", padding_mode=padding_mode, align_corners=bool(align_corners))
    sampled = sampled.reshape(batch_size, num_views, channels, *sample_shape)
    if valid_mask is not None:
        sampled = sampled * valid_mask.to(sampled.dtype)[:, :, None]
    return sampled.contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("sample_features_by_grid_v2", _sample_features_by_grid_v2_impl)


def _sample_features_by_grid_v2_meta(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
    valid_mask: torch.Tensor = None,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    if feature_map.dim() == 4:
        return torch.empty((feature_map.shape[0], feature_map.shape[1], grid.shape[1]), device="meta", dtype=feature_map.dtype)
    if grid.dim() == 4:
        shape = (feature_map.shape[0], feature_map.shape[1], feature_map.shape[2], grid.shape[2])
    else:
        shape = (feature_map.shape[0], feature_map.shape[1], feature_map.shape[2], grid.shape[2], grid.shape[3])
    return torch.empty(shape, device="meta", dtype=feature_map.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("sample_features_by_grid_v2", _sample_features_by_grid_v2_meta)


def sample_features_by_grid_v2(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
    valid_mask: torch.Tensor = None,
    align_corners: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return torch.ops.fusionrepo.sample_features_by_grid_v2(
        feature_map,
        grid,
        valid_mask,
        bool(align_corners),
        str(padding_mode),
    )
