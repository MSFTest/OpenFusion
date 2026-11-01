# -*- coding: utf-8 -*-
"""EPNet++ point detector adapter module."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class PointRPNAdapterModule(nn.Module):
    """
    调用冻结真实 3D detector，并构造 EPNet++ 点级输入特征。

    Args:
        batch: dict，至少包含 `points`。真实 detector 可使用 `pcd_path`。

    Returns:
        dict:
            detector_prediction: boxes_3d/scores_3d/labels_3d
            point_features: (B, 3, N)，默认使用 xyz 作为可训练融合输入。
    """

    def __init__(self, point_cloud_detector: nn.Module):
        super().__init__()
        self.point_cloud_detector = point_cloud_detector

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        points = batch["points"]
        if points.dim() == 2:
            points = points.unsqueeze(0)
        with torch.no_grad():
            prediction = self.point_cloud_detector(batch)
        return {
            "detector_prediction": prediction,
            "point_features": points[..., :3].transpose(1, 2).contiguous(),
            "points_batched": points,
        }
