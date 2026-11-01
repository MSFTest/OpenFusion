# -*- coding: utf-8 -*-
"""DeepInteraction 可学习 BEV 位置编码。"""

from torch import nn


class PositionEmbeddingLearned(nn.Module):
    """以 ``2→C→C`` Conv1d 编码 BEV cell-center 坐标。

    Args:
        channels: 输出通道数，正式配置 128。
    Returns:
        输入 ``position[B,N,2]`` 输出 ``[B,C,N]``；dtype/device 跟随模块。
    原源码：``decoder_utils.py:16-32``。
    """
    def __init__(self, channels=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(2, channels, 1), nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True), nn.Conv1d(channels, channels, 1),
        )

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 ``position_embedding_head`` checkpoint 键。"""
        return {
            key.replace("position_embedding_head.", "layers.", 1): value
            for key, value in state_dict.items()
        }

    def forward(self, position):
        return self.layers(position.transpose(1, 2).contiguous())
