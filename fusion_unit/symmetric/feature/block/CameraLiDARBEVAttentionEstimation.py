# -*- coding: utf-8 -*-
"""
CameraLiDARBEVAttentionEstimation

3D-CVF 私有 Camera-LiDAR BEV 注意力权重估计模块。

该模块保留原始 3D-CVF 的 BEV 级两模态 softmax 注意力逻辑：
相机 BEV 特征与 LiDAR BEV 特征拼接后，经 1x1 Conv 网络输出两路权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion_unit.symmetric.feature.block.CrossModalGateEstimator import CrossModalGateEstimator


class CameraLiDARBEVAttentionEstimation(nn.Module):
    """
    Camera-LiDAR BEV 注意力权重估计模块。

    Args:
        channels:
            单路 BEV 特征通道数 C。

    输入:
        camera_bev_features:
            相机 BEV 特征，shape 为 (B, C, H, W)。

        lidar_bev_features:
            LiDAR BEV 特征，shape 为 (B, C, H, W)。

    输出:
        attention_weights:
            跨模态注意力权重，shape 为 (B, 2, H, W)，第 0 通道为相机权重，
            第 1 通道为 LiDAR 权重。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attention = CrossModalGateEstimator(
            in_channels=int(channels),
            num_inputs=2,
            hidden_channels=int(channels),
            out_channels=2,
            layout="BCHW",
            activation="softmax_modal",
        )

    def forward(
        self,
        camera_bev_features: torch.Tensor,
        lidar_bev_features: torch.Tensor,
    ) -> torch.Tensor:
        if camera_bev_features.shape != lidar_bev_features.shape:
            raise ValueError("camera_bev_features and lidar_bev_features must have the same shape")
        return self.attention([camera_bev_features, lidar_bev_features])
