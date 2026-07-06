# -*- coding: utf-8 -*-
"""
normalize_grouped_points_with_centroid

说明：
    对分组点云按组计算有效点质心，并将每个有效点平移到以组质心为原点。
    可用于 frustum、voxel、ROI 或 instance grouped points 的局部坐标归一化。

Args:
    grouped_points: (G, M, 3) 或 (B, G, M, 3)
        分组后的点坐标。

    grouped_mask: (G, M) 或 (B, G, M)
        有效点掩码，True 表示真实点，False 表示 padding。

Returns:
    normalized_points: 与 grouped_points 形状一致
        减去组内有效点质心后的点坐标，padding 位置置零。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _normalize_grouped_points_with_centroid_impl(
    grouped_points: torch.Tensor,
    grouped_mask: torch.Tensor,
) -> torch.Tensor:
    single_batch = grouped_points.dim() == 3
    if single_batch:
        grouped_points = grouped_points.unsqueeze(0)
        grouped_mask = grouped_mask.unsqueeze(0)

    if grouped_points.dim() != 4 or grouped_points.shape[-1] != 3:
        raise ValueError("grouped_points must have shape (G,M,3) or (B,G,M,3)")
    if grouped_mask.shape != grouped_points.shape[:-1]:
        raise ValueError("grouped_mask shape must match grouped_points without channel dim")

    mask_f = grouped_mask.to(dtype=grouped_points.dtype)
    counts = mask_f.sum(dim=2).clamp_min(1.0)
    flat_points = grouped_points.reshape(-1, grouped_points.shape[2], 3)
    flat_counts = counts.reshape(-1)
    centroids = torch.ops.fusionrepo.compute_voxel_centroid(flat_points, flat_counts)
    centroids = centroids.reshape(grouped_points.shape[0], grouped_points.shape[1], 1, 3)

    normalized = (grouped_points - centroids) * mask_f.unsqueeze(-1)
    if single_batch:
        normalized = normalized.squeeze(0)
    return normalized


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("normalize_grouped_points_with_centroid", _normalize_grouped_points_with_centroid_impl)


def _normalize_grouped_points_with_centroid_meta(
    grouped_points: torch.Tensor,
    grouped_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(grouped_points, device="meta")


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("normalize_grouped_points_with_centroid", _normalize_grouped_points_with_centroid_meta)


def normalize_grouped_points_with_centroid(
    grouped_points: torch.Tensor,
    grouped_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.normalize_grouped_points_with_centroid(
        grouped_points,
        grouped_mask,
    )
