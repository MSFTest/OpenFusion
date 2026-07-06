# -*- coding: utf-8 -*-
"""
BEVfusion 相机 frustum 特征到 BEV 平面池化 wrapper。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _bevfusion_camera_bev_pooling_impl(
    feat_3d: torch.Tensor,
    bev_indices: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    """
    Args:
        feat_3d: (B, D, C, H, W)
            相机 frustum 三维特征。
        bev_indices: (B, D, H, W, 2)
            每个 frustum 位置对应的 BEV 网格索引 (i, j)，负数表示无效。
        bev_h: int
            BEV 网格高度。
        bev_w: int
            BEV 网格宽度。

    Returns:
        bev_feat: (B, C, bev_h, bev_w)
            sum pooling 后的相机 BEV 特征。
    """
    if feat_3d.dim() != 5:
        raise ValueError("feat_3d must have shape (B, D, C, H, W)")
    if bev_indices.dim() != 5 or bev_indices.shape[-1] != 2:
        raise ValueError("bev_indices must have shape (B, D, H, W, 2)")
    b, d, c, h, w = feat_3d.shape
    if bev_indices.shape[:4] != (b, d, h, w):
        raise ValueError("bev_indices shape must match feat_3d batch/depth/spatial dims")

    features = feat_3d.permute(0, 1, 3, 4, 2).reshape(b * d * h * w, c)
    idx = bev_indices.reshape(b, d * h * w, 2).to(device=feat_3d.device, dtype=torch.long)
    batch = torch.arange(b, device=feat_3d.device, dtype=torch.long).view(b, 1).expand(b, d * h * w)
    z = torch.zeros_like(batch)
    coords = torch.stack(
        [batch.reshape(-1), z.reshape(-1), idx[..., 0].reshape(-1), idx[..., 1].reshape(-1)],
        dim=1,
    ).to(dtype=torch.int32)
    return torch.ops.fusionrepo.voxel_features_to_bev_scatter(
        features,
        coords,
        b,
        int(bev_h),
        int(bev_w),
        "sum",
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("bevfusion_camera_bev_pooling", _bevfusion_camera_bev_pooling_impl)


def _bevfusion_camera_bev_pooling_meta(
    feat_3d: torch.Tensor,
    bev_indices: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    return torch.empty(
        (feat_3d.shape[0], feat_3d.shape[2], int(bev_h), int(bev_w)),
        device="meta",
        dtype=feat_3d.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("bevfusion_camera_bev_pooling", _bevfusion_camera_bev_pooling_meta)


def bevfusion_camera_bev_pooling(
    feat_3d: torch.Tensor,
    bev_indices: torch.Tensor,
    bev_h: int,
    bev_w: int,
) -> torch.Tensor:
    return torch.ops.fusionrepo.bevfusion_camera_bev_pooling(
        feat_3d,
        bev_indices,
        int(bev_h),
        int(bev_w),
    )
