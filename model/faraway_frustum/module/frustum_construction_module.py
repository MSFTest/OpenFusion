# -*- coding: utf-8 -*-
"""Frustum 构建与远距离目标判定模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import utils  # noqa: F401
import common  # noqa: F401
import fusion_unit  # noqa: F401


class FrustumConstructionModule(nn.Module):
    """组合 2D 实例与相机坐标点云，生成 Faraway-Frustum 点簇。

    Args:
        max_points: 每个实例最多保留的点数 P。
        histogram_bins: 逐轴直方图聚类 bin 数。

    Inputs:
        points_2d_img: ``(N, 2)``，像素坐标 ``(u, v)``。
        points_3d_cam0: ``(N, 3)``，相机 rect 坐标系点 ``(x, y, z)``。
        boxes_2d: ``(K, 4)``，图像框 ``(x1, y1, x2, y2)``。
        masks_2d: ``(K, H, W)``，实例 mask；空张量时退化为 box frustum。
        labels: ``(K,)``，类别索引，顺序必须与 detector metainfo 一致。
        depth_thresholds: ``(C,)``，相机 z 方向远距离阈值。

    Returns:
        dict，包含 ``clusters (K,P,3)``、``counts (K,)``、
        ``centroids (K,3)``、``is_faraway (K,)``。
    """

    def __init__(self, max_points: int = 2048, histogram_bins: int = 20):
        super().__init__()
        self.max_points = int(max_points)
        self.histogram_bins = int(histogram_bins)

    def forward(
        self,
        points_2d_img: torch.Tensor,
        points_3d_cam0: torch.Tensor,
        boxes_2d: torch.Tensor,
        masks_2d: torch.Tensor,
        labels: torch.Tensor,
        depth_thresholds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        clusters, counts = torch.ops.fusionrepo.faraway_mask_frustum_pointcloud_crop(
            points_2d_img,
            points_3d_cam0,
            boxes_2d,
            masks_2d,
            self.max_points,
        )
        centroids = torch.ops.fusionrepo.faraway_frustum_histogram_clustering(
            clusters,
            counts,
            self.histogram_bins,
        )
        safe_labels = labels.long().clamp(min=0, max=max(int(depth_thresholds.numel()) - 1, 0))
        is_faraway = torch.ops.fusionrepo.faraway_object_decision(
            centroids,
            safe_labels,
            depth_thresholds,
        )
        is_faraway = is_faraway & (counts > 0)
        return {
            "clusters": clusters,
            "counts": counts,
            "centroids": centroids,
            "is_faraway": is_faraway,
        }
