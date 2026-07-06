# -*- coding: utf-8 -*-
"""
Image Feature Interface

用于统一调用图像特征提取器，使 AVOD 融合网络
能够无缝接入不同来源的图像 backbone（如 mmdet）。
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ImageFeatureInterface(nn.Module):
    """
    图像特征提取接口。

    输入:
        images: (B, 3, H, W)

    输出:
        image_features: (B, C, Hf, Wf)

    参数:
        extractor: 外部图像特征提取器
        feat_index: 若 extractor 返回多层特征 (list/tuple)，选择哪一层
    """

    def __init__(self, extractor: nn.Module, feat_index: int = 0):
        super().__init__()
        self.extractor = extractor
        self.feat_index = feat_index

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        if images.dim() != 4:
            raise ValueError("images must be (B,3,H,W)")

        feats = self.extractor(images)

        # 兼容 mmdet / FPN 等返回多层特征
        if isinstance(feats, (list, tuple)):
            feats = feats[self.feat_index]

        if not torch.is_tensor(feats):
            raise TypeError("extractor must return Tensor or list/tuple of Tensor")

        if feats.dim() != 4:
            raise ValueError("image feature must be (B,C,H,W)")

        return feats