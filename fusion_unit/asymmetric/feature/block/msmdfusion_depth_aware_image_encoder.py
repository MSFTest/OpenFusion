"""多尺度深度感知图像压缩。原源码：mmdet3d/models/detectors/MSMDFusion.py，
MSMDFusionDetector.__init__ 107–124 行及 depth_aware_channel_compression 335–369 行。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class DepthAwareImageEncoder(nn.Module):
    """三组独立 Conv/BN/ReLU，参数与最终配置的 FPN 256 通道一致。

    Args: feats 为 FPN 前三级 [B*6,256,H_i,W_i]；depth 为 [B*6,1,H,W]
    的原图像像素深度图。Returns: 三张 [B*6,49,H_i,W_i] 图像特征；
    前两层 5x5，第三层 3x3，BN eps=0.001、momentum=0.01。
    原实现稍后以 [0,0,1,2] 顺序复用输出；该复用由调用者完成。
    """

    def __init__(self):
        super().__init__()
        self.conv1x1_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(257, 49, 5, padding=2, bias=False),
                          nn.BatchNorm2d(49, eps=0.001, momentum=0.01), nn.ReLU()),
            nn.Sequential(nn.Conv2d(257, 49, 5, padding=2, bias=False),
                          nn.BatchNorm2d(49, eps=0.001, momentum=0.01), nn.ReLU()),
            nn.Sequential(nn.Conv2d(257, 49, 3, padding=1, bias=False),
                          nn.BatchNorm2d(49, eps=0.001, momentum=0.01), nn.ReLU()),
        ])

    def forward(self, feats: list[torch.Tensor], depth: torch.Tensor) -> list[torch.Tensor]:
        """原源码 360–369 行：默认双线性插值，无 align_corners 参数。"""
        outputs = []
        for i in range(3):
            sparse_depth = F.interpolate(depth, feats[i].shape[-2:], mode='bilinear')
            outputs.append(self.conv1x1_blocks[i](feature_concatenation([feats[i], sparse_depth], 1)))
        return outputs
