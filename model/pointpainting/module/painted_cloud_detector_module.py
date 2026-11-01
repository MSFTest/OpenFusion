# -*- coding: utf-8 -*-
"""
PaintedCloudDetectorModule

调用点云接口处理 painted point cloud。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PaintedCloudDetectorModule(nn.Module):
    """
    painted points 点云检测模块。

    输入:
        painted_points: Tensor 或 list[Tensor]
        batch: 原始 batch 字典

    输出:
        点云接口返回的 dict。
    """

    def __init__(self, point_cloud_backbone: nn.Module):
        super().__init__()
        self.point_cloud_backbone = point_cloud_backbone
        self.detector_input_channels = 4 if hasattr(point_cloud_backbone, "extractor") else None

    def forward(self, painted_points, batch: dict | None = None) -> dict:
        batch = batch or {}
        if torch.is_tensor(painted_points):
            points_input = [painted_points]
        else:
            points_input = list(painted_points)
        if self.detector_input_channels is not None:
            points_input = [
                pts[:, : self.detector_input_channels].contiguous()
                if torch.is_tensor(pts) and pts.shape[-1] > self.detector_input_channels
                else pts
                for pts in points_input
            ]
        return self.point_cloud_backbone(
            points=points_input,
            img_metas=batch.get("img_metas", None),
        )
