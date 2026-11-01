# -*- coding: utf-8 -*-
"""BEVPoolV2 ranks 与 interval 准备算子。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library


def _prepare_bev_pool_indices_impl(
    coor: torch.Tensor,
    grid_lower_bound: torch.Tensor,
    grid_interval: torch.Tensor,
    grid_size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 ego 坐标离散为 BEV voxel，并生成 BEVPoolV2 排序索引。

    Args:
        coor: ego 坐标，shape 为 ``(B,N,D,H,W,3)``。
        grid_lower_bound: 网格下界，shape 为 ``(3,)``。
        grid_interval: 网格间隔，shape 为 ``(3,)``。
        grid_size: 网格尺寸 ``(X,Y,Z)``，shape 为 ``(3,)``。

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: 依次为
        ``ranks_bev``、``ranks_depth``、``ranks_feat``、
        ``interval_starts``、``interval_lengths``，均为一维 long 张量。
    """
    bsz, num_cams, depth_bins, feat_h, feat_w, _ = coor.shape
    num_points = bsz * num_cams * depth_bins * feat_h * feat_w
    ranks_depth = torch.arange(num_points, dtype=torch.long, device=coor.device)
    ranks_feat = torch.arange(num_points // depth_bins, dtype=torch.long, device=coor.device)
    ranks_feat = ranks_feat.reshape(bsz, num_cams, 1, feat_h, feat_w)
    ranks_feat = ranks_feat.expand(bsz, num_cams, depth_bins, feat_h, feat_w).flatten()
    voxel = ((coor - grid_lower_bound.to(coor).view(1, 1, 1, 1, 1, 3)) /
             grid_interval.to(coor).view(1, 1, 1, 1, 1, 3)).long().view(num_points, 3)
    batch_idx = torch.arange(bsz, dtype=torch.long, device=coor.device)
    batch_idx = batch_idx.view(bsz, 1).expand(bsz, num_points // bsz).reshape(num_points)
    kept = (
        (voxel[:, 0] >= 0) & (voxel[:, 0] < grid_size[0].long())
        & (voxel[:, 1] >= 0) & (voxel[:, 1] < grid_size[1].long())
        & (voxel[:, 2] >= 0) & (voxel[:, 2] < grid_size[2].long())
    )
    voxel = voxel[kept]
    ranks_depth = ranks_depth[kept]
    ranks_feat = ranks_feat[kept]
    batch_idx = batch_idx[kept]
    ranks_bev = batch_idx * (grid_size[2].long() * grid_size[1].long() * grid_size[0].long())
    ranks_bev = ranks_bev + voxel[:, 2] * (grid_size[1].long() * grid_size[0].long())
    ranks_bev = ranks_bev + voxel[:, 1] * grid_size[0].long() + voxel[:, 0]
    order = ranks_bev.argsort()
    ranks_bev = ranks_bev[order].long()
    ranks_depth = ranks_depth[order].long()
    ranks_feat = ranks_feat[order].long()
    if ranks_bev.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=coor.device)
        return empty, empty.clone(), empty.clone(), empty.clone(), empty.clone()
    interval_mask = torch.ones_like(ranks_bev, dtype=torch.bool)
    interval_mask[1:] = ranks_bev[1:] != ranks_bev[:-1]
    starts = torch.where(interval_mask)[0].long()
    lengths = torch.empty_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = ranks_bev.shape[0] - starts[-1]
    return ranks_bev, ranks_depth, ranks_feat, starts, lengths


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("prepare_bev_pool_indices", _prepare_bev_pool_indices_impl)


def _prepare_bev_pool_indices_meta(coor, grid_lower_bound, grid_interval, grid_size):
    upper = coor.numel() // 3
    return tuple(torch.empty((upper,), dtype=torch.long, device="meta") for _ in range(5))


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("prepare_bev_pool_indices", _prepare_bev_pool_indices_meta)


def prepare_bev_pool_indices(coor, grid_lower_bound, grid_interval, grid_size):
    """调用 ``fusionrepo::prepare_bev_pool_indices``。"""
    return torch.ops.fusionrepo.prepare_bev_pool_indices(
        coor, grid_lower_bound, grid_interval, grid_size
    )
