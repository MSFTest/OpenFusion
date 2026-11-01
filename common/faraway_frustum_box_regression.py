# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有 BEV 三维框参数回归模块。"""

from __future__ import annotations

import torch
import torch.nn as nn

import torch.nn.functional as F


class _DepthwiseSeparableBlock(nn.Module):
    """MobileNet 风格深度可分离卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.depthwise_bn(self.depthwise(x)), inplace=True)
        x = F.relu6(self.pointwise_bn(self.pointwise(x)), inplace=True)
        return x


class FarawayFrustumBoxRegression(nn.Module):
    """从单通道 Frustum BEV 密度图回归原项目式七维三维框参数。

    Args:
        无初始化参数。

    Inputs:
        bev_features: ``(B, 1, H, W)``，每个远距离实例的 BEV 密度图。
        默认使用当前 FusionRepo Faraway-Frustum 推荐局部分辨率
        ``H=300, W=200``，对应 ``30m x 20m``、``0.1m`` 栅格。

    Returns:
        raw_outputs: ``(B, 8)``，前六维为
        ``(x_shifted, y, z_shifted, height, width, length)``，最后两维为
        ``(sin(2 * rotation_y), cos(2 * rotation_y))`` 的双角连续表示。
        三维框绕竖直轴旋转 ``pi`` 后几何等价，双角表示可避免车辆前后
        朝向的双峰分布被回归为错误的中间角度。
        ``x_shifted`` 与 ``z_shifted`` 是相对 frustum centroid 并加上正偏移的
        坐标，便于训练稳定收敛。
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.features = nn.Sequential(
            _DepthwiseSeparableBlock(16, 32, stride=1),
            _DepthwiseSeparableBlock(32, 64, stride=2),
            _DepthwiseSeparableBlock(64, 64, stride=1),
            _DepthwiseSeparableBlock(64, 128, stride=2),
            _DepthwiseSeparableBlock(128, 128, stride=1),
            _DepthwiseSeparableBlock(128, 256, stride=2),
            _DepthwiseSeparableBlock(256, 256, stride=1),
            _DepthwiseSeparableBlock(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 8)
        with torch.no_grad():
            self.fc.bias.copy_(
                torch.tensor([75.0, 1.5, 76.5, 1.5, 1.6, 4.0, 0.0, 1.0])
            )

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """执行 BEV 特征提取、全局池化和框参数回归。"""
        if bev_features.dim() != 4 or bev_features.shape[1] != 1:
            raise ValueError("bev_features must have shape (B, 1, H, W)")
        features = self.features(self.stem(bev_features.float()))
        features = torch.flatten(self.pool(features), 1)
        return self.fc(self.dropout(features))
