# -*- coding: utf-8 -*-
"""Original-style EPNet image fusion module."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils  # noqa: F401 - registers fusionrepo operators
from model.epnet.module.li_fusion_module import LIFusionModule


class _ImageBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OriginalStyleImageFusionModule(nn.Module):
    """
    复刻 EPNet 原项目的多尺度图像分支融合形式。

    原项目在 PointNet++ SA 多级点特征处执行 Img_Block、Feature_Gather、IA_Layer
    注意力融合，并在末端将多尺度 deconv 图像特征拼接后再次采样回点特征。本模块在
    FusionRepo 中保留同样的数据流形态：多级图像卷积、逐点采样、逐点门控融合、末端
    deconv 拼接融合。PointNet++ SA 本体由冻结 mmdet3d detector 承担，新增参数只训练
    图像-点融合适配层。

    Inputs:
        images: (B, 3, H, W)，原始 KITTI 图像张量，像素坐标与 uv 对齐。
        point_features: (B, C, N)，点特征。
        uv: (B, N, 2)，原始图像像素坐标。
        valid_mask: (B, N)，投影有效点。

    Returns:
        dict:
            original_style_features: (B, C, N)，多尺度原论文式融合特征。
            original_style_weight: (B, C, N)，末端融合权重。

    调用算子:
        fusionrepo::epnet_grid_generator、fusionrepo::epnet_image_sampler，以及 LIFusionModule
        内部的 fusionrepo LI-Fusion 算子。
    """

    def __init__(
        self,
        point_channels: int = 128,
        image_channels: List[int] | None = None,
        deconv_reduce: int = 16,
    ):
        super().__init__()
        if image_channels is None:
            image_channels = [3, 64, 128, 256, 512]
        self.image_blocks = nn.ModuleList(
            [_ImageBlock(image_channels[i], image_channels[i + 1]) for i in range(len(image_channels) - 1)]
        )
        self.stage_align = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(image_channels[i + 1], point_channels, 1, bias=False),
                    nn.BatchNorm1d(point_channels),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(image_channels) - 1)
            ]
        )
        self.stage_fusion = nn.ModuleList([LIFusionModule(point_channels) for _ in range(len(image_channels) - 1)])
        self.deconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(image_channels[i + 1], deconv_reduce, kernel_size=2 ** (i + 1), stride=2 ** (i + 1))
                for i in range(len(image_channels) - 1)
            ]
        )
        self.image_fusion = nn.Sequential(
            nn.Conv2d(deconv_reduce * (len(image_channels) - 1), point_channels // 4, 1, bias=False),
            nn.BatchNorm2d(point_channels // 4),
            nn.ReLU(inplace=True),
        )
        self.final_align = nn.Sequential(
            nn.Conv1d(point_channels // 4, point_channels, 1, bias=False),
            nn.BatchNorm1d(point_channels),
            nn.ReLU(inplace=True),
        )
        self.final_fusion = LIFusionModule(point_channels)

    @staticmethod
    def _sample(feature_map: torch.Tensor, uv: torch.Tensor, image_hw) -> torch.Tensor:
        depth = torch.ones(uv.shape[:2], dtype=feature_map.dtype, device=feature_map.device)
        _, grid, _ = torch.ops.fusionrepo.epnet_grid_generator(int(image_hw[1]), int(image_hw[0]), uv, depth)
        return torch.ops.fusionrepo.epnet_image_sampler(feature_map, grid)

    def forward(
        self,
        images: torch.Tensor,
        point_features: torch.Tensor,
        uv: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_hw = (int(images.shape[-2]), int(images.shape[-1]))
        x = images / 255.0
        point = point_features
        pyramid = []
        for block, align, fusion, deconv in zip(self.image_blocks, self.stage_align, self.stage_fusion, self.deconvs):
            x = block(x)
            sampled = self._sample(x, uv, image_hw)
            sampled = align(sampled)
            point = fusion(point, sampled)["fused_features"]
            pyramid.append(deconv(x))

        target_hw = pyramid[0].shape[-2:]
        resized = [p if p.shape[-2:] == target_hw else F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False) for p in pyramid]
        image_fused = self.image_fusion(torch.cat(resized, dim=1))
        sampled_final = self.final_align(self._sample(image_fused, uv, image_hw))
        sampled_final = sampled_final * valid_mask.unsqueeze(1).to(dtype=sampled_final.dtype)
        final = self.final_fusion(point, sampled_final)
        return {
            "original_style_features": final["fused_features"],
            "original_style_weight": final["fusion_weight"],
        }
