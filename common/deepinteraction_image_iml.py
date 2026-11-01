# -*- coding: utf-8 -*-
"""DeepInteraction 图像模态内局部交互层。"""

from torch import nn

from common.deepinteraction_local_context_attention import LocalContextAttention


class ImageIML(nn.Module):
    """逐相机对 ``[B,V,C,H,W]`` 图像特征自注意并返回相同布局。

    Args:
        channels: 通道数，正式配置 128。
        kernel_size: 局部窗口，正式配置 9。
    原源码：``deepinteraction_encoder.py:16,31`` 的 ``I_IML``。
    """
    def __init__(self, channels=128, kernel_size=9):
        super().__init__()
        self.local_attention = LocalContextAttention(channels, kernel_size)

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 ``I_IML`` 的直接 LocalContextAttentionBlock 权重。"""
        mapped = LocalContextAttention.remap_original_state_dict(state_dict)
        return {"local_attention." + key: value for key, value in mapped.items()}

    def forward(self, image_feature):
        batch, views, channels, height, width = image_feature.shape
        flat = image_feature.reshape(batch * views, channels, height, width)
        return self.local_attention(flat, flat).reshape(
            batch, views, channels, height, width
        )
