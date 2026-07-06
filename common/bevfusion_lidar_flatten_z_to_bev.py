# -*- coding: utf-8 -*-
"""
BEVfusion LiDAR 三维体素特征高度维压缩 wrapper。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _bevfusion_lidar_flatten_z_to_bev_impl(feat_3d: torch.Tensor) -> torch.Tensor:
    """
    Args:
        feat_3d: (B, C, Z, H, W)
            LiDAR 三维体素特征体。

    Returns:
        bev_feat: (B, C, H, W)
            沿 Z 维 sum pooling 后的 BEV 特征。
    """
    if feat_3d.dim() != 5:
        raise ValueError("feat_3d must have shape (B, C, Z, H, W)")
    return torch.ops.fusionrepo.reduce_pool(feat_3d, 2, "sum", False)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("bevfusion_lidar_flatten_z_to_bev", _bevfusion_lidar_flatten_z_to_bev_impl)


def _bevfusion_lidar_flatten_z_to_bev_meta(feat_3d: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (feat_3d.shape[0], feat_3d.shape[1], feat_3d.shape[3], feat_3d.shape[4]),
        device="meta",
        dtype=feat_3d.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("bevfusion_lidar_flatten_z_to_bev", _bevfusion_lidar_flatten_z_to_bev_meta)


def bevfusion_lidar_flatten_z_to_bev(feat_3d: torch.Tensor) -> torch.Tensor:
    return torch.ops.fusionrepo.bevfusion_lidar_flatten_z_to_bev(feat_3d)
