# -*- coding: utf-8 -*-
"""
FrustumFCNModule

功能：
    沿 frustum 序列执行 1D FCN 编码、上采样和多尺度对齐拼接融合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.frustum_conv_block import FrustumConvBlock
from common.frustum_deconv_1d import FrustumDeconv1D
from common.shared_mlp_1d import SharedMLP1D


class FrustumFCNModule(nn.Module):
    """
    Frustum 方向 FCN 融合模块。

    Inputs:
        sequence_features: (P, C, T)

    Returns:
        fused_sequence_features: (P, 768, T_ref)
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 3):
        super().__init__()
        self.num_classes = int(num_classes)

        def conv(c_in: int, c_out: int, kernel: int, stride: int = 1, padding: int = 0):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            )

        self.block1_conv1 = conv(in_channels + self.num_classes, 128, 3, 1, 1)
        self.block2_conv1 = conv(128, 128, 3, 2, 1)
        self.block2_conv2 = conv(128, 128, 3, 1, 1)
        self.block2_merge = conv(128 + 128 + self.num_classes, 128, 1)
        self.block3_conv1 = conv(128, 256, 3, 2, 1)
        self.block3_conv2 = conv(256, 256, 3, 1, 1)
        self.block3_merge = conv(256 + 256 + self.num_classes, 256, 1)
        self.block4_conv1 = conv(256, 512, 3, 2, 1)
        self.block4_conv2 = conv(512, 512, 3, 1, 1)
        self.block4_merge = conv(512 + 512 + self.num_classes, 512, 1)

        self.block2_deconv = FrustumDeconv1D(128, 256, kernel_size=1, stride=1, padding=0)
        self.block3_deconv = FrustumDeconv1D(256, 256, kernel_size=2, stride=2, padding=0)
        self.block4_deconv = FrustumDeconv1D(512, 256, kernel_size=4, stride=4, padding=0)
        self.out_channels = 768

    def forward(self, multiscale_features: list[torch.Tensor], proposal_labels: torch.Tensor) -> torch.Tensor:
        features = [item.transpose(1, 2).contiguous() for item in multiscale_features]
        one_hot = F.one_hot(
            proposal_labels.clamp(0, self.num_classes - 1).long(), num_classes=self.num_classes
        ).to(features[0])
        conditioned = [
            torch.cat([feature, one_hot.unsqueeze(-1).expand(-1, -1, feature.shape[-1])], dim=1)
            for feature in features
        ]
        x = self.block1_conv1(conditioned[0])
        x = self.block2_conv2(self.block2_conv1(x))
        x2 = self.block2_merge(torch.cat([x, conditioned[1]], dim=1))
        x = self.block3_conv2(self.block3_conv1(x2))
        x3 = self.block3_merge(torch.cat([x, conditioned[2]], dim=1))
        x = self.block4_conv2(self.block4_conv1(x3))
        x4 = self.block4_merge(torch.cat([x, conditioned[3]], dim=1))

        y2 = self.block2_deconv(x2)
        y3 = self.block3_deconv(x3)
        y4 = self.block4_deconv(x4)
        fused = torch.ops.fusionrepo.align_and_concat_multiscale_1d([y2, y3, y4])
        return fused
