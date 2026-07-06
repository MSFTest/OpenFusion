# -*- coding: utf-8 -*-
"""
CameraBEVFeatureAlignment

3D-CVF 私有相机 BEV 特征对齐模块。

该模块保留原始 3D-CVF 行为：对相机 BEV 序列特征预测逐位置门控权重，
并将该权重乘回原特征，用于缓解投影后相机 BEV 特征的空间偏差。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.shared_mlp_1d import SharedMLP1D


class CameraBEVFeatureAlignment(nn.Module):
    """
    相机 BEV 特征门控对齐模块。

    Args:
        channels:
            输入相机 BEV 特征通道数 C。

    输入:
        bev_camera_features:
            shape 为 (B, C, M)，M 为 BEV 网格数量。

    输出:
        aligned_bev_features:
            shape 为 (B, C, M)，与输入形状一致。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.offset_predictor = SharedMLP1D(
            in_channels=int(channels),
            channels=[int(channels), 2],
            use_bn=True,
            activation=True,
            activate_last=False,
            input_layout="BCL",
            output_layout="BCL",
        )

    def forward(self, bev_camera_features: torch.Tensor) -> torch.Tensor:
        if bev_camera_features.dim() != 3:
            raise ValueError("bev_camera_features must have shape (B, C, M)")

        offsets = self.offset_predictor(bev_camera_features)
        align_weight = torch.sigmoid(offsets[:, :1, :])
        return bev_camera_features * align_weight
