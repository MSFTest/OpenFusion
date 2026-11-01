# -*- coding: utf-8 -*-
"""
PaintedPointFeatureModule

PointPainting 辅助点级特征建模模块。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.fc_classification_regression_head import FCClassificationRegressionHead
from common.pointpainting_painted_point_backbone import PaintedPointBackbone
from common.pointpainting_painted_point_feature_encoder import PaintedPointFeatureEncoder


class PaintedPointFeatureModule(nn.Module):
    """
    使用已落地的网络型算子对 painted points 做辅助点级特征建模。

    输入:
        painted_points: (N, in_channels)

    输出:
        dict:
            point_features: (N, 256)
            cls_scores: (N, num_classes)
            box_preds: (N, box_dim)
    """

    def __init__(self, in_channels: int, num_classes: int = 3, box_dim: int = 7):
        super().__init__()
        self.encoder = PaintedPointFeatureEncoder(in_channels)
        self.backbone = PaintedPointBackbone()
        self.head = FCClassificationRegressionHead(
            in_channels=256,
            hidden_channels=256,
            num_classes=num_classes,
            box_dim=box_dim,
            num_shared_layers=1,
        )

    def forward(self, painted_points: torch.Tensor) -> dict:
        encoded = self.encoder(painted_points)
        point_features = self.backbone(encoded)
        cls_scores, box_preds = self.head(point_features)
        return {
            "point_features": point_features,
            "cls_scores": cls_scores,
            "box_preds": box_preds,
        }
