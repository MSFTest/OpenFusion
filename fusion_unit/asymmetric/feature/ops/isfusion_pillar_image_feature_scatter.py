# -*- coding: utf-8 -*-
"""
IS-Fusion pillar 图像特征聚合并散射到 BEV 的补全算子。

Args:
    sampled_features: 图像采样特征，shape 为 (B, V, C, P, L)。
    pillar_coors: pillar 坐标，shape 为 (B, P, 4) 或 (P_total, 4)，顺序为 (batch, z, y, x)。
    bev_size: 方形 BEV 网格边长。

Returns:
    Tensor: 图像 BEV 特征，shape 为 (B, C, bev_size, bev_size)。
"""

from __future__ import annotations

import torch
from torch.library import Library

import utils.voxel_features_to_bev_scatter  # noqa: F401


def _isfusion_pillar_image_feature_scatter_impl(
    sampled_features: torch.Tensor,
    pillar_coors: torch.Tensor,
    bev_size: int,
) -> torch.Tensor:
    if sampled_features.dim() != 5:
        raise ValueError("sampled_features must have shape (B,V,C,P,L)")
    batch_size, _, channels, num_pillars, _ = sampled_features.shape
    per_pillar = sampled_features.sum(dim=1).sum(dim=-1)

    if pillar_coors.dim() == 3:
        coords = pillar_coors[:, :num_pillars].reshape(batch_size * num_pillars, 4)
        features = per_pillar.permute(0, 2, 1).reshape(batch_size * num_pillars, channels)
    elif pillar_coors.dim() == 2:
        coords_list = []
        features_list = []
        for batch_idx in range(batch_size):
            mask = pillar_coors[:, 0].long() == batch_idx
            coords_b = pillar_coors[mask]
            coords_list.append(coords_b)
            features_list.append(per_pillar[batch_idx, :, : coords_b.shape[0]].transpose(0, 1))
        coords = torch.cat(coords_list, dim=0) if coords_list else pillar_coors.reshape(0, 4)
        features = torch.cat(features_list, dim=0) if features_list else per_pillar.new_empty((0, channels))
    else:
        raise ValueError("pillar_coors must have shape (B,P,4) or (P_total,4)")

    return torch.ops.fusionrepo.voxel_features_to_bev_scatter(
        features.contiguous(),
        coords.to(torch.long).contiguous(),
        int(batch_size),
        int(bev_size),
        int(bev_size),
        "sum",
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("isfusion_pillar_image_feature_scatter", _isfusion_pillar_image_feature_scatter_impl)


def _isfusion_pillar_image_feature_scatter_meta(
    sampled_features: torch.Tensor,
    pillar_coors: torch.Tensor,
    bev_size: int,
) -> torch.Tensor:
    return torch.empty((sampled_features.shape[0], sampled_features.shape[2], int(bev_size), int(bev_size)), device="meta", dtype=sampled_features.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("isfusion_pillar_image_feature_scatter", _isfusion_pillar_image_feature_scatter_meta)


def isfusion_pillar_image_feature_scatter(
    sampled_features: torch.Tensor,
    pillar_coors: torch.Tensor,
    bev_size: int,
) -> torch.Tensor:
    return torch.ops.fusionrepo.isfusion_pillar_image_feature_scatter(sampled_features, pillar_coors, int(bev_size))
