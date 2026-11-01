# -*- coding: utf-8 -*-
"""DeepInteraction LiDAR 到图像局部注意力融合块。"""

from torch import nn

from common.deepinteraction_local_context_attention import LocalContextAttention


class LiDARToImageAttention(nn.Module):
    """以图像为 target、反投影 LiDAR 为 source 做局部交互。

    输入/输出均为 ``[B,V,C,H,W]``，内部按 ``B*V`` 展平；dtype/device 与
    输入一致。Args 为 channels（正式 128）和 kernel_size（正式 9）。
    原源码：``encoder_utils.py:202-213`` ``MMRI_P2I.Local``。
    """
    def __init__(self, channels=128, kernel_size=9):
        super().__init__()
        self.local_attention = LocalContextAttention(channels, kernel_size)

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 ``MMRI_P2I.Local`` checkpoint 键。"""
        stripped = {
            (key[len("Local."):] if key.startswith("Local.") else key): value
            for key, value in state_dict.items()
        }
        mapped = LocalContextAttention.remap_original_state_dict(stripped)
        return {"local_attention." + key: value for key, value in mapped.items()}

    def forward(self, old_image, warped_lidar):
        batch, views, channels, height, width = old_image.shape
        result = self.local_attention(
            old_image.reshape(batch * views, channels, height, width),
            warped_lidar.reshape(batch * views, channels, height, width),
        )
        return result.reshape(batch, views, channels, height, width)
