# -*- coding: utf-8 -*-
"""
FeatureFusion

说明：
    该文件实现"3D/2D 特征融合"的 VirConv 私有模块。

功能：
    融合 NRConv 3D 与 NRConv 2D 两条分支输出的体素特征。

    该模块只负责：
    - 多路特征通道拼接
    - 线性映射与特征融合

    不负责：
    - 特征生成
    - 归一化方式选择

结构：
    torch.cat([feat_3d, feat_2d], dim=1)
        -> Linear
        -> BatchNorm1d
        -> ReLU
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """
    3D/2D 特征融合模块。

    该模块用于融合 NRConv 3D 与 NRConv 2D 两条分支输出的体素特征，
    将来自三维空间与图像平面的互补信息整合为统一的体素级表示。

    输入：
        feat_3d: (M, C_3d)
            来自 NRConv 3D 的体素特征。

        feat_2d: (M, C_2d)
            来自 NRConv 2D 的体素特征。

    输出：
        fused: (M, C_out)
            融合后的体素特征。
    """

    def __init__(
        self,
        in_channels_3d: int,
        in_channels_2d: int,
        out_channels: int
    ):
        super().__init__()

        in_channels = in_channels_3d + in_channels_2d
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        feat_3d: torch.Tensor,
        feat_2d: torch.Tensor
    ) -> torch.Tensor:
        if feat_3d.shape[0] != feat_2d.shape[0]:
            raise ValueError("feat_3d and feat_2d must have the same number of voxels")

        x = torch.cat([feat_3d, feat_2d], dim=1)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x