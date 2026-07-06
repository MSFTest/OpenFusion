# -*- coding: utf-8 -*-
"""
fc_classification_regression_head

通用 FC 分类回归检测头。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FCClassificationRegressionHead(nn.Module):
    """
    共享全连接特征变换 + 分类分支 + 回归分支模块。

    Args:
        in_channels:
            输入特征维度 C_in。

        hidden_channels:
            共享全连接层隐藏维度。

        num_classes:
            分类分支输出类别数。

        box_dim:
            回归分支输出框参数维度。

        num_shared_layers:
            共享全连接层数量，至少为 1。

    输入:
        features:
            二维特征张量，shape 为 (N, C_in)。

    输出:
        cls_scores:
            分类得分，shape 为 (N, num_classes)。

        box_preds:
            框回归预测，shape 为 (N, box_dim)。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        box_dim: int = 7,
        num_shared_layers: int = 1,
    ):
        super().__init__()
        if num_shared_layers < 1:
            raise ValueError("num_shared_layers must be >= 1")

        layers: list[nn.Module] = []
        c_in = int(in_channels)
        for _ in range(int(num_shared_layers)):
            layers.append(nn.Linear(c_in, int(hidden_channels)))
            layers.append(nn.ReLU(inplace=True))
            c_in = int(hidden_channels)

        self.shared_fc = nn.Sequential(*layers)
        self.cls_head = nn.Linear(int(hidden_channels), int(num_classes))
        self.reg_head = nn.Linear(int(hidden_channels), int(box_dim))

    def forward(self, features: torch.Tensor):
        if features.dim() != 2:
            raise ValueError("features must have shape (N, C)")
        shared = self.shared_fc(features)
        return self.cls_head(shared), self.reg_head(shared)
