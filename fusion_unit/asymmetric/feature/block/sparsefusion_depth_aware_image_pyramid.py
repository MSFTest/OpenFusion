# -*- coding: utf-8 -*-
"""SparseFusion 稀疏深度感知图像金字塔。"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from common.sparsefusion_layers import BasicBlock, initialize_like_sparsefusion
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class SparseFusionDepthAwareImagePyramid(nn.Module):
    """逐级把 LiDAR 稀疏深度注入四尺度图像特征。

    原源码 ``mmdet3d/models/utils/depth_encoder.py::DepthEncoderResNet``。
    输入 depth ``[B*V,2,H/4,W/4]`` 与四层 image feature
    ``[B*V,256,Hl,Wl]``；正式输出四个 ``[B*V,128,Hl,Wl]``。每级先执行
    BasicBlock（stride 1/2/2/2），再按通道拼接图像并用 3×3 Conv 融合。
    ``level_pos`` 只加到 clone 输出，不反馈到下一尺度。
    """

    def __init__(self, depth_channels=2, image_channels=256, hidden_channels=128, level_count=4):
        super().__init__()
        self.conv_depth = nn.Sequential(nn.Conv2d(depth_channels, hidden_channels, 3, padding=1, bias=True),
                                        nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.layers = nn.ModuleList([nn.Sequential(BasicBlock(hidden_channels, 1 if i == 0 else 2))
                                     for i in range(level_count)])
        self.fuse_layers = nn.ModuleList([nn.Conv2d(image_channels + hidden_channels, hidden_channels, 3, padding=1)
                                          for _ in range(level_count)])
        self.level_pos = nn.Parameter(torch.zeros(level_count, hidden_channels))
        initialize_like_sparsefusion(self)
        nn.init.normal_(self.level_pos)

    def forward(self, sparse_depth: torch.Tensor, image_features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(image_features) != len(self.layers):
            raise ValueError("图像金字塔层数不匹配")
        depth, outputs = self.conv_depth(sparse_depth), []
        for level, image in enumerate(image_features):
            depth = self.layers[level](depth)
            if depth.shape[0] != image.shape[0] or depth.shape[-2:] != image.shape[-2:]:
                raise ValueError(f"第 {level} 层深度与图像空间 shape 不一致")
            depth = self.fuse_layers[level](feature_concatenation([depth, image], 1))
            outputs.append((depth + self.level_pos[level].view(1, -1, 1, 1)).clone())
        return outputs
