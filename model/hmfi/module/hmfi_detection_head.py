# -*- coding: utf-8 -*-
"""
HMFIDetectionHead

HMFI检测头：使用ROI特征收集 + 分类/回归头进行目标检测

流程：
    1. gather_roi_features: 从融合体素特征中收集每个ROI内的特征
    2. Mean Pooling: 将ROI内特征聚合
    3. Shared FC: 共享特征变换
    4. Classification Head + Regression Head
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple


class HMFIDetectionHead(nn.Module):
    """
    HMFI 检测头

    使用 gather_roi_features 从融合体素特征中收集 ROI 内特征，
    然后通过 Mean Pooling + Shared FC + 分类/回归头进行检测。

    Args:
        in_channels: 输入特征通道数
        num_classes: 类别数，默认1（车辆检测）
        box_dim: 回归参数维度，默认7 (x, y, z, dx, dy, dz, yaw)
        grid_size: ROI内网格大小，默认6
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        box_dim: int = 7,
        grid_size: int = 6,
        hidden_channels: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.box_dim = box_dim
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels

        # 共享特征变换
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, box_dim)
        )

    def forward(
        self,
        rois: torch.Tensor,
        voxel_coords: torch.Tensor,
        fused_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rois: (K, 7) ROI参数 (x, y, z, dx, dy, dz, yaw)
            voxel_coords: (N, 3) 体素坐标
            fused_features: (N, C) 融合后的体素特征

        Returns:
            {
                "cls_score": (K, num_classes),
                "reg_offset": (K, box_dim),
                "gathered_features": (K, N, C),
                "valid_counts": (K,)
            }
        """
        # 收集ROI内特征
        gathered_features, valid_counts = torch.ops.fusionrepo.gather_roi_features(
            rois,
            voxel_coords,
            fused_features
        )  # (K, N, C), (K,)

        # gather_roi_features keeps valid features at their original voxel
        # indices and writes zeros elsewhere. valid_counts is a count, not a
        # compact prefix length, so prefix masks would discard most valid voxels.
        summed = gathered_features.sum(dim=1)
        pooled_feat = summed / valid_counts.clamp(min=1).unsqueeze(-1).to(gathered_features.dtype)

        # Shared feature transformation
        feat = self.shared_fc(pooled_feat)  # (K, hidden)

        # Classification and regression
        cls_score = self.cls_head(feat)  # (K, num_classes)
        reg_offset = self.reg_head(feat)  # (K, box_dim)

        return {
            "cls_score": cls_score,
            "reg_offset": reg_offset,
            "gathered_features": gathered_features,
            "valid_counts": valid_counts,
            "pooled_feat": pooled_feat
        }
