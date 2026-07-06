# -*- coding: utf-8 -*-
"""
camera_lidar_bev_feature_fusion

说明：
    3D-CVF Camera-LiDAR BEV 特征融合适配算子。

    该算子保留原始三输入接口，内部将两路 BEV 特征堆叠，并将
    attention_weights 转换为可广播权重后调用通用加权融合算子。

Args:
    camera_bev_features:
        相机 BEV 特征，shape 为 (B, C, H, W)。

    lidar_bev_features:
        LiDAR BEV 特征，shape 为 (B, C, H, W)。

    attention_weights:
        两路跨模态注意力权重，shape 为 (B, 2, H, W)。
        第 0 通道对应相机权重，第 1 通道对应 LiDAR 权重。

Returns:
    fused_bev_features:
        融合后的 BEV 特征，shape 为 (B, C, H, W)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _camera_lidar_bev_feature_fusion_impl(
    camera_bev_features: torch.Tensor,
    lidar_bev_features: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    if camera_bev_features.shape != lidar_bev_features.shape:
        raise ValueError("camera_bev_features and lidar_bev_features must have the same shape")
    if attention_weights.dim() != 4 or attention_weights.shape[1] != 2:
        raise ValueError("attention_weights must have shape (B, 2, H, W)")
    if attention_weights.shape[0] != camera_bev_features.shape[0]:
        raise ValueError("attention_weights batch size must match feature batch size")
    if attention_weights.shape[2:] != camera_bev_features.shape[2:]:
        raise ValueError("attention_weights spatial size must match feature spatial size")

    features = torch.stack([camera_bev_features, lidar_bev_features], dim=0)
    weights = attention_weights.permute(1, 0, 2, 3).unsqueeze(2).contiguous()
    return torch.ops.fusionrepo.feature_fusion_broadcast_weighted_sum(features, weights)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "camera_lidar_bev_feature_fusion",
    _camera_lidar_bev_feature_fusion_impl,
)


def _camera_lidar_bev_feature_fusion_meta(
    camera_bev_features: torch.Tensor,
    lidar_bev_features: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        tuple(camera_bev_features.shape),
        device="meta",
        dtype=camera_bev_features.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "camera_lidar_bev_feature_fusion",
    _camera_lidar_bev_feature_fusion_meta,
)


def camera_lidar_bev_feature_fusion(
    camera_bev_features: torch.Tensor,
    lidar_bev_features: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.camera_lidar_bev_feature_fusion(
        camera_bev_features,
        lidar_bev_features,
        attention_weights,
    )
