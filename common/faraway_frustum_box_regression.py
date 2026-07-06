# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有 BEV 三维框参数回归模块。"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class FarawayFrustumBoxRegression(nn.Module):
    """从单通道 Frustum BEV 密度图回归五维三维框参数。

    Args:
        无初始化参数。

    Inputs:
        bev_features: ``(B, 1, H, W)``，每个远距离实例的 BEV 密度图。

    Returns:
        regression_outputs: ``(B, 5)``，依次为
            ``(delta_z, width, length, height, alpha)``。
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2dNormActivationBlock(1, 16, 3, 1, use_bn=False)
        self.conv2 = Conv2dNormActivationBlock(16, 32, 3, 1, use_bn=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """执行 BEV 特征提取、全局池化和框参数回归。"""
        if bev_features.dim() != 4 or bev_features.shape[1] != 1:
            raise ValueError("bev_features must have shape (B, 1, H, W)")
        features = self.conv2(self.conv1(bev_features))
        features = torch.flatten(self.pool(features), 1)
        return self.fc2(self.relu(self.fc1(features)))
