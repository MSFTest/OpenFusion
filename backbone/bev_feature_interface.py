# -*- coding: utf-8 -*-
"""
BEV Feature Interface

用于统一调用 BEV 特征提取器，使 AVOD 融合网络
能够接入不同来源的 BEV backbone。
"""

from __future__ import annotations
import torch
import torch.nn as nn


class BEVFeatureInterface(nn.Module):
    """
    BEV 特征提取接口。

    输入:
        bev_input: 任意 BEV backbone 所需输入

    输出:
        bev_features: (B, C, Hb, Wb)

    参数:
        extractor: 外部 BEV 特征提取器
        feat_index: 若 extractor 返回多层特征，选择哪一层
    """

    def __init__(self, extractor: nn.Module, feat_index: int = 0):
        super().__init__()
        self.extractor = extractor
        self.feat_index = feat_index

    def forward(self, bev_input):

        feats = self.extractor(bev_input)

        if isinstance(feats, (list, tuple)):
            feats = feats[self.feat_index]

        if not torch.is_tensor(feats):
            raise TypeError("extractor must return Tensor or list/tuple of Tensor")

        if feats.dim() != 4:
            raise ValueError("bev feature must be (B,C,H,W)")

        return feats