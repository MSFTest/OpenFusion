# -*- coding: utf-8 -*-
"""EPNet++ lightweight image stream module."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class ImageStreamModule(nn.Module):
    """
    生成 EPNet++ 图像流融合特征与图像 confidence map。

    Args:
        image_features: Tensor，shape 为 (B, C_in, H, W)，通常来自冻结 mmdet FPN。

    Returns:
        dict:
            image_grid_features: (B, C_out, H, W)
            image_conf_map: (B, 1, H, W)
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 128):
        super().__init__()
        self.proj = Conv2dNormActivationBlock(in_channels, out_channels, kernel_size=3)
        self.refine = Conv2dNormActivationBlock(out_channels, out_channels, kernel_size=3)
        self.conf = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.refine(self.proj(image_features))
        return {
            "image_grid_features": x,
            "image_conf_map": self.conf(x),
        }
