# -*- coding: utf-8 -*-
"""DeepInteraction proposal-conditioned DynamicConv。"""

import torch.nn.functional as F
from torch import nn


class DynamicConv(nn.Module):
    """用 query 生成两组 C×C 参数，连续调制 ROI token。

    Args:
        channels: query/ROI 通道，正式配置 128。
        roi_size: 正方形 ROI 边长，正式配置 7。
    Returns:
        proposal_feature ``[1,K,C]`` 与 roi_feature ``[S*S,K,C]`` 输出
        ``[1,K,C]``。保留两级 bmm、LayerNorm-ReLU、flatten、线性映射顺序。
    原源码：``decoder_utils.py:584-629``。
    """
    def __init__(self, channels=128, roi_size=7):
        super().__init__()
        self.channels = int(channels)
        self.num_parameters = self.channels * self.channels
        self.dynamic_layer = nn.Linear(channels, 2 * self.num_parameters)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.out_layer = nn.Linear(channels * roi_size * roi_size, channels)
        self.norm3 = nn.LayerNorm(channels)

    def forward(self, proposal_feature, roi_feature):
        feature = roi_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(proposal_feature).permute(1, 0, 2)
        first = parameters[:, :, :self.num_parameters].reshape(
            -1, self.channels, self.channels
        )
        second = parameters[:, :, self.num_parameters:].reshape(
            -1, self.channels, self.channels
        )
        feature = F.relu(self.norm1(feature.bmm(first)))
        feature = F.relu(self.norm2(feature.bmm(second)))
        return F.relu(self.norm3(self.out_layer(feature.flatten(1))))
