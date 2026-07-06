# -*- coding: utf-8 -*-
"""
BEVfusion 相机 FPN 多尺度特征融合模块。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVFusionCameraFPNFusion(nn.Module):
    """
    多尺度相机特征对齐与加和融合。

    Args:
        in_channels_list: Sequence[int]
            各尺度输入特征通道数。
        out_channels: int
            输出统一通道数。
        use_bn: bool
            是否在 1x1 Conv 后使用 BatchNorm2d。
        use_relu: bool
            是否使用 ReLU 激活。

    Inputs:
        feats: list[Tensor]
            多尺度特征列表，每个元素 shape 为 (B, C_i, H_i, W_i)。

    Returns:
        fused: (B, C_out, H_ref, W_ref)
            对齐到首个尺度空间尺寸后的加和融合特征。
    """

    def __init__(
        self,
        in_channels_list,
        out_channels: int,
        use_bn: bool = True,
        use_relu: bool = True,
    ):
        super().__init__()
        self.proj_layers = nn.ModuleList()
        for in_channels in in_channels_list:
            layers = [
                nn.Conv2d(
                    int(in_channels),
                    int(out_channels),
                    kernel_size=1,
                    bias=not bool(use_bn),
                )
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(int(out_channels)))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            self.proj_layers.append(nn.Sequential(*layers))

    def forward(self, feats) -> torch.Tensor:
        if len(feats) != len(self.proj_layers):
            raise ValueError("feats length must match projection layer count")
        ref_h, ref_w = feats[0].shape[-2:]
        fused = None
        for feat, proj in zip(feats, self.proj_layers):
            x = proj(feat)
            if x.shape[-2:] != (ref_h, ref_w):
                x = F.interpolate(x, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            fused = x if fused is None else fused + x
        return fused


CameraFPNFusion = BEVFusionCameraFPNFusion
