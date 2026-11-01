# -*- coding: utf-8 -*-
"""DeepInteraction 局部上下文注意力通用层。"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class LocalContextAttention(nn.Module):
    """原 CUDA localattention 的数学等价 PyTorch 实现。

    Args:
        channels: query/key/value 通道数，正式配置为 128。
        kernel_size: 奇数局部窗口，正式配置为 9。
        last_affine: value 投影 BatchNorm 是否具有可学习 affine。
    Returns:
        输入 ``target_feature/source_feature [B,C,H,W]`` 对应的
        ``[B,C,H,W]`` 输出，保持 dtype/device/NCHW 布局并支持自动求导。

    保留两级 query/key 1x1 Conv-BN-ReLU、一级 value 投影、Xavier 初始化、
    零 padding、通道点积、``softmax(sim/sqrt(C))`` 和加权顺序。复用
    ``common.Conv2dNormActivationBlock``；原 checkpoint 的 ``conv/norm`` 键
    可通过 :meth:`remap_original_state_dict` 显式映射。原源码：
    ``encoder_utils.py:36-135`` 及 ``locatt_ops/*.cu``。
    """

    def __init__(self, channels: int = 128, kernel_size: int = 9,
                 last_affine: bool = True):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = int(kernel_size)
        self.query_project = nn.Sequential(
            Conv2dNormActivationBlock(channels, channels, 1),
            Conv2dNormActivationBlock(channels, channels, 1),
        )
        self.key_project = nn.Sequential(
            Conv2dNormActivationBlock(channels, channels, 1),
            Conv2dNormActivationBlock(channels, channels, 1),
        )
        self.value_project = Conv2dNormActivationBlock(channels, channels, 1)
        if not last_affine:
            norm = self.value_project.block[1]
            norm.affine = False
            norm.register_parameter("weight", None)
            norm.register_parameter("bias", None)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

    @staticmethod
    def remap_original_state_dict(state_dict):
        """把原 ``ConvBN.conv/norm`` 键映射到复用块的 ``block.0/1`` 键。"""
        mapped = {}
        for key, value in state_dict.items():
            key = (key.replace(".conv.", ".block.0.")
                   .replace(".norm.", ".block.1.")
                   .replace(".bn.", ".block.1."))
            if key.startswith("value_project.conv."):
                key = key.replace("value_project.conv.", "value_project.block.0.")
            if key.startswith("value_project.norm."):
                key = key.replace("value_project.norm.", "value_project.block.1.")
            if key.startswith("value_project.bn."):
                key = key.replace("value_project.bn.", "value_project.block.1.")
            mapped[key] = value
        return mapped

    def forward(self, target_feature: torch.Tensor,
                source_feature: torch.Tensor) -> torch.Tensor:
        query = self.query_project(target_feature)
        key = self.key_project(source_feature)
        value = self.value_project(source_feature)
        batch, channels, height, width = query.shape
        windows = self.kernel_size ** 2
        key_window = F.unfold(key, self.kernel_size, padding=self.kernel_size // 2)
        key_window = key_window.reshape(batch, channels, windows, height * width)
        similarity = (query.reshape(batch, channels, 1, height * width)
                      * key_window).sum(1).transpose(1, 2)
        weight = F.softmax(similarity / math.sqrt(channels), dim=-1)
        value_window = F.unfold(value, self.kernel_size,
                                padding=self.kernel_size // 2)
        value_window = value_window.reshape(batch, channels, windows, height * width)
        output = (value_window * weight.transpose(1, 2).unsqueeze(1)).sum(2)
        return output.reshape(batch, channels, height, width)
