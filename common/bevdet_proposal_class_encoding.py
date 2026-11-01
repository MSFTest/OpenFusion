# -*- coding: utf-8 -*-
"""BEVDet proposal 类别编码模块。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalClassEncoding(nn.Module):
    """
    将 proposal 类别 one-hot 线性映射后注入 LiDAR query。

    Args:
        query_channels: query 特征通道数。
        num_classes: 检测类别数。

    Inputs:
        lidar_query: LiDAR query，shape 为 ``(B,C,K)``。
        proposal_classes: 类别索引，shape 为 ``(B,K)``。

    Returns:
        Tensor: 类别增强后的 query，shape 为 ``(B,C,K)``。
    """

    def __init__(self, query_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_encoding = nn.Conv1d(self.num_classes, int(query_channels), kernel_size=1)

    def forward(self, lidar_query: torch.Tensor, proposal_classes: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(
            proposal_classes.long(), num_classes=self.num_classes
        ).to(lidar_query.dtype)
        encoded = torch.ops.fusionrepo.linear_projection(
            one_hot,
            self.class_encoding.weight.squeeze(-1),
            self.class_encoding.bias,
        ).transpose(1, 2)
        return lidar_query + encoded
