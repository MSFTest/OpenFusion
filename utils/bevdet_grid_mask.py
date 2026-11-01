# -*- coding: utf-8 -*-
"""DAL 图像主干使用的 GridMask 数据增强层。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BEVDetGridMask(nn.Module):
    """按原 DAL 参数在归一化图像上施加共享网格遮罩。

    Args:
        ratio: 每个网格周期的遮挡比例。
        mode: ``1`` 表示反转基础网格遮罩。
        prob: 训练时启用增强的概率。

    Inputs:
        images: 归一化图像，shape 为 ``(B*N,C,H,W)``。

    Returns:
        Tensor: shape 不变的增强图像；评估模式保持输入不变。
    """

    def __init__(self, ratio: float = 0.5, mode: int = 1, prob: float = 0.7):
        super().__init__()
        self.ratio = float(ratio)
        self.mode = int(mode)
        self.prob = float(prob)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.rand() > self.prob or min(images.shape[-2:]) <= 2:
            return images
        height, width = images.shape[-2:]
        mask_h, mask_w = int(1.5 * height), int(1.5 * width)
        period = int(np.random.randint(2, height))
        line = min(max(int(period * self.ratio + 0.5), 1), period - 1)
        mask = images.new_ones((mask_h, mask_w))
        start_h, start_w = int(np.random.randint(period)), int(np.random.randint(period))
        for index in range(mask_h // period):
            start = period * index + start_h
            mask[start:min(start + line, mask_h), :] = 0
        for index in range(mask_w // period):
            start = period * index + start_w
            mask[:, start:min(start + line, mask_w)] = 0
        top, left = (mask_h - height) // 2, (mask_w - width) // 2
        mask = mask[top:top + height, left:left + width]
        if self.mode == 1:
            mask = 1.0 - mask
        return images * mask.view(1, 1, height, width)
