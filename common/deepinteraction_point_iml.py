# -*- coding: utf-8 -*-
"""DeepInteraction LiDAR 模态内局部交互层。"""

from torch import nn

from common.deepinteraction_local_context_attention import LocalContextAttention


class PointIML(nn.Module):
    """对 ``[B,C,H,W]`` LiDAR-BEV 自注意，输出同 shape/dtype/device。

    Args:
        channels: 通道数，正式配置 128。
        kernel_size: 局部窗口，正式配置 9。
    原源码：``deepinteraction_encoder.py:11,26`` 的 ``P_IML``。
    """
    def __init__(self, channels=128, kernel_size=9):
        super().__init__()
        self.local_attention = LocalContextAttention(channels, kernel_size)

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 ``P_IML`` 的直接 LocalContextAttentionBlock 权重。"""
        mapped = LocalContextAttention.remap_original_state_dict(state_dict)
        return {"local_attention." + key: value for key, value in mapped.items()}

    def forward(self, lidar_feature):
        return self.local_attention(lidar_feature, lidar_feature)
