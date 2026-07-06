# -*- coding: utf-8 -*-
"""
centerfusion_radar_heatmap_generate

说明：
    CenterFusion 私有雷达热力图生成算子。将目标级雷达物理特征写回
    2D 特征图中心点位置，生成与图像特征空间对齐的雷达特征热力图。

Args:
    radar_feats: (N, 3)
        目标级雷达特征，格式为 (depth, vx, vy)。
    centers_2d: (N, 2)
        特征图坐标系中的目标中心点，格式为 (x, y)。
    feat_hw: (2,)
        特征图尺寸，格式为 (H, W)。

Returns:
    heatmap: (3, H, W)
        雷达特征热力图。若多个目标落在同一中心，后写入目标覆盖先写入目标。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _centerfusion_radar_heatmap_generate_impl(
    radar_feats: torch.Tensor,
    centers_2d: torch.Tensor,
    feat_hw: torch.Tensor,
) -> torch.Tensor:
    if radar_feats.dim() != 2 or radar_feats.shape[1] != 3:
        raise ValueError("radar_feats must have shape (N, 3)")
    if centers_2d.dim() != 2 or centers_2d.shape[1] != 2:
        raise ValueError("centers_2d must have shape (N, 2)")
    if feat_hw.numel() != 2:
        raise ValueError("feat_hw must contain (H, W)")

    H = int(feat_hw[0].item())
    W = int(feat_hw[1].item())
    heatmap = torch.zeros((3, H, W), device=radar_feats.device, dtype=radar_feats.dtype)

    xs = centers_2d[:, 0].long().clamp(0, W - 1)
    ys = centers_2d[:, 1].long().clamp(0, H - 1)

    for i in range(radar_feats.shape[0]):
        heatmap[:, ys[i], xs[i]] = radar_feats[i]

    return heatmap


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("centerfusion_radar_heatmap_generate", _centerfusion_radar_heatmap_generate_impl)


def _centerfusion_radar_heatmap_generate_meta(
    radar_feats: torch.Tensor,
    centers_2d: torch.Tensor,
    feat_hw: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((3, int(feat_hw[0]), int(feat_hw[1])), device="meta", dtype=radar_feats.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("centerfusion_radar_heatmap_generate", _centerfusion_radar_heatmap_generate_meta)


def centerfusion_radar_heatmap_generate(
    radar_feats: torch.Tensor,
    centers_2d: torch.Tensor,
    feat_hw: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.centerfusion_radar_heatmap_generate(
        radar_feats,
        centers_2d,
        feat_hw,
    )
