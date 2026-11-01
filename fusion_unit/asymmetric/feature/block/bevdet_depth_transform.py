# -*- coding: utf-8 -*-
"""BEVDet 图像特征与 LiDAR 深度先验的非对称融合模块。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class DepthTransformNet(nn.Module):
    """
    融合图像特征和可选 LiDAR 投影深度，预测 LSS 深度概率与变换特征。

    Args:
        in_channels: 图像特征通道数。
        depth_bins: 深度离散数量 D。
        out_channels: BEV pooling 图像特征通道数。
        hidden_channels: 融合隐藏通道数，默认等于 ``in_channels``。
        use_lidar_depth: 是否使用 LiDAR 深度先验。

    Inputs:
        image_feats: 图像特征，shape 为 ``(B,N,C,H,W)``。
        lidar_depth: 可选深度图，shape 为 ``(B,N,Hd,Wd)`` 或 ``(B,N,1,Hd,Wd)``。

    Returns:
        Tuple[Tensor, Tensor]: 深度概率 ``(B,N,D,H,W)`` 与变换特征
        ``(B,N,Cout,H,W)``。
    """

    def __init__(
        self,
        in_channels: int,
        depth_bins: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        use_lidar_depth: bool = True,
        image_downsample: int = 8,
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.depth_bins = int(depth_bins)
        self.out_channels = int(out_channels)
        self.use_lidar_depth = bool(use_lidar_depth)
        self.image_downsample = int(image_downsample)
        self.lidar_depth_encoder = None
        input_channels = int(in_channels)
        if self.use_lidar_depth:
            self.lidar_depth_encoder = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 32, kernel_size=5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=5, stride=int(2 * self.image_downsample / 8), padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            input_channels += 64
        depth_first = Conv2dNormActivationBlock(
            input_channels, hidden_channels, 3, use_bn=True, activation=True
        )
        depth_second = Conv2dNormActivationBlock(
            hidden_channels, hidden_channels, 3, use_bn=True, activation=True
        )
        self.depth_net = nn.Sequential(
            *depth_first.block.children(),
            *depth_second.block.children(),
            nn.Conv2d(hidden_channels, self.depth_bins + self.out_channels, kernel_size=1),
        )

    def forward(
        self, image_feats: torch.Tensor, lidar_depth: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image_feats.dim() != 5:
            raise ValueError("image_feats must have shape (B,N,C,H,W)")
        bsz, num_cams, channels, feat_h, feat_w = image_feats.shape
        x = image_feats.reshape(bsz * num_cams, channels, feat_h, feat_w)
        if self.use_lidar_depth:
            if lidar_depth is None:
                lidar_depth = x.new_zeros((bsz, num_cams, 1, feat_h, feat_w))
            if lidar_depth.dim() == 4:
                lidar_depth = lidar_depth.unsqueeze(2)
            prior = lidar_depth.reshape(bsz * num_cams, 1, *lidar_depth.shape[-2:]).to(x.dtype)
            prior = self.lidar_depth_encoder(prior)
            if prior.shape[-2:] != (feat_h, feat_w):
                raise RuntimeError(
                    "LiDAR depth encoder output %s does not match image feature %s"
                    % (tuple(prior.shape[-2:]), (feat_h, feat_w))
                )
            x = torch.cat([x, prior], dim=1)
        prediction = self.depth_net(x)
        depth_prob = prediction[:, :self.depth_bins].softmax(dim=1)
        tran_feat = prediction[:, self.depth_bins:self.depth_bins + self.out_channels]
        return (
            depth_prob.view(bsz, num_cams, self.depth_bins, feat_h, feat_w),
            tran_feat.view(bsz, num_cams, self.out_channels, feat_h, feat_w),
        )
