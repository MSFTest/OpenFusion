# -*- coding: utf-8 -*-
"""
feature_fusion_broadcast_weighted_sum

说明：
    多路同形状特征的可广播权重加权求和融合算子。

    该算子用于相机、LiDAR、雷达、伪点云等多源特征已经空间或索引对齐后，
    按路级标量权重、点级权重、ROI 级权重或 BEV 空间动态权重执行融合。

Args:
    features:
        待融合特征，shape 为 (K, ...)。K 表示特征路数。
        例如 (2, B, C, H, W)、(2, B, C, N)、(3, K_roi, G, C)。

    weights:
        可选融合权重。支持 None、(K,) 或任意可广播到 features 的形状。
        例如 3D-CVF 可传 (2, B, 1, H, W)，表示 BEV 空间动态权重。

Returns:
    fused:
        融合后的特征，shape 为 features 去掉第一维后的形状。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _feature_fusion_broadcast_weighted_sum_impl(
    features: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if features.dim() < 1:
        raise ValueError("features must have shape (K, ...)")
    if features.shape[0] <= 0:
        raise ValueError("features first dimension K must be greater than 0")

    if weights is None:
        return features.sum(dim=0)

    weights = weights.to(device=features.device, dtype=features.dtype)
    if weights.dim() == 1:
        if weights.shape[0] != features.shape[0]:
            raise ValueError("weights length must match features.shape[0]")
        weights = weights.view((weights.shape[0],) + (1,) * (features.dim() - 1))
    elif weights.shape[0] != features.shape[0]:
        raise ValueError("weights first dimension must match features.shape[0]")

    return (features * weights).sum(dim=0)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "feature_fusion_broadcast_weighted_sum",
    _feature_fusion_broadcast_weighted_sum_impl,
)


def _feature_fusion_broadcast_weighted_sum_meta(
    features: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        tuple(features.shape[1:]),
        device="meta",
        dtype=features.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "feature_fusion_broadcast_weighted_sum",
    _feature_fusion_broadcast_weighted_sum_meta,
)


def feature_fusion_broadcast_weighted_sum(
    features: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_fusion_broadcast_weighted_sum(
        features,
        weights,
    )
