# -*- coding: utf-8 -*-
"""
BEVfusion Camera/LiDAR BEV 特征拼接 wrapper。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _bevfusion_bev_concat_fusion_impl(
    bev_cam: torch.Tensor,
    bev_lidar: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Args:
        bev_cam: (B, C_cam, H, W)
            相机 BEV 特征。
        bev_lidar: (B, C_lidar, H, W)
            LiDAR BEV 特征。
        dim: int
            拼接维度，默认通道维。

    Returns:
        bev_fused: (B, C_cam + C_lidar, H, W)
            拼接后的 BEV 融合特征。
    """
    return torch.ops.fusionrepo.feature_concatenation([bev_cam, bev_lidar], int(dim))


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("bevfusion_bev_concat_fusion", _bevfusion_bev_concat_fusion_impl)


def _bevfusion_bev_concat_fusion_meta(
    bev_cam: torch.Tensor,
    bev_lidar: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    shape = list(bev_cam.shape)
    shape[int(dim)] = bev_cam.shape[int(dim)] + bev_lidar.shape[int(dim)]
    return torch.empty(tuple(shape), device="meta", dtype=bev_cam.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("bevfusion_bev_concat_fusion", _bevfusion_bev_concat_fusion_meta)


def bevfusion_bev_concat_fusion(
    bev_cam: torch.Tensor,
    bev_lidar: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    return torch.ops.fusionrepo.bevfusion_bev_concat_fusion(bev_cam, bev_lidar, int(dim))
