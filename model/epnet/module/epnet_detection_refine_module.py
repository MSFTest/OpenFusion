# -*- coding: utf-8 -*-
"""Detection refinement module for EPNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class EPNetDetectionRefineModule(nn.Module):
    """
    用融合点级证据修正冻结 3D detector 的检测置信度。

    Inputs:
        detector_boxes: (M, 7)，LiDAR 坐标 3D 框。
        detector_scores: (M,)，真实 detector 输出分数。
        detector_labels: (M,)，真实 detector 输出类别索引。
        point_cls_logits: (B, N)，融合头点级前景 logits。

    Returns:
        dict:
            boxes_3d: (M, 7)，保持 detector 框坐标语义。
            scores_3d: (M,)，由 detector score 与融合点证据共同得到。
            labels_3d: (M,)，类别索引。
    """

    def __init__(self, min_scale: float = 0.5):
        super().__init__()
        self.min_scale = float(min_scale)

    def forward(
        self,
        detector_boxes: torch.Tensor,
        detector_scores: torch.Tensor,
        detector_labels: torch.Tensor,
        point_cls_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if detector_scores.numel() == 0:
            return {
                "boxes_3d": detector_boxes,
                "scores_3d": detector_scores,
                "labels_3d": detector_labels,
            }
        point_score = torch.sigmoid(point_cls_logits).mean().clamp(0.0, 1.0)
        scale = self.min_scale + (1.0 - self.min_scale) * point_score
        return {
            "boxes_3d": detector_boxes,
            "scores_3d": detector_scores * scale,
            "labels_3d": detector_labels,
        }
