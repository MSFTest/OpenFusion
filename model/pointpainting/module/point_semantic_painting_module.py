# -*- coding: utf-8 -*-
"""
PointSemanticPaintingModule

根据投影像素坐标查询语义分数，并将语义分数拼接到点云特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import fusion_unit  # noqa: F401
import utils  # noqa: F401


class PointSemanticPaintingModule(nn.Module):
    """
    PointPainting 语义 painting 模块。

    输入:
        points_lidar: (N, D) 原始 LiDAR 点云特征。
        pixel_coords: (N, 2) 投影像素坐标。
        semantic_score_map: (H, W, C) 或 (B, H, W, C) 语义分数图。
        valid_mask: (N,) 有效投影点 mask。

    输出:
        painted_points: (M, D + C)，M 为有效投影点数量。
    """

    def forward(
        self,
        points_lidar: torch.Tensor,
        pixel_coords: torch.Tensor,
        semantic_score_map: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        batch_index: int = 0,
    ) -> torch.Tensor:
        if semantic_score_map.dim() == 4:
            semantic_score_map = semantic_score_map[int(batch_index)]
        if valid_mask is None:
            valid_mask = torch.ones((points_lidar.shape[0],), device=points_lidar.device, dtype=torch.bool)

        valid_points = points_lidar[valid_mask]
        valid_pixels = pixel_coords[valid_mask]
        point_scores = torch.ops.fusionrepo.pointpainting_point_semantic_query(
            valid_pixels,
            semantic_score_map.to(device=points_lidar.device, dtype=points_lidar.dtype),
        )
        return torch.ops.fusionrepo.pointpainting_point_feature_concatenation(
            valid_points,
            point_scores,
        )
