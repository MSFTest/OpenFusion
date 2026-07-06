# -*- coding: utf-8 -*-
"""
feature_fusion_weighted_sum

说明：
    将“多路特征逐元素加权求和融合”注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.feature_fusion_weighted_sum(features, weights)

兼容性：
    - 兼容 PyTorch 2.0.1
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    对多路同形状特征做逐元素加权求和融合。

数学形式：
    fused = sum_i weights[i] * features[i]

输入：
    features: (K, ...)
        K 路待融合特征，要求每一路特征形状相同。
        例如：
            (2, N, C)
            (2, B, C, H, W)
            (3, K, G, C)

    weights: (K,) 或 None
        每一路特征的融合权重。
        如果为 None，则默认每一路权重都为 1。
        如果希望实现平均融合，可在外部传入：
            [1/K, 1/K, ..., 1/K]

输出：
    fused: (...)
        融合后的特征，形状与单路特征相同。

等价替代：
    - avod::anchor_feature_fusion        -> 两路平均融合
    - cvf::camera_lidar_bev_feature_fusion -> 两路加权融合
    - logonet::merge_fusion_features     -> 多路直接相加
"""

from __future__ import annotations

import torch
from torch.library import Library




# ============================================================
# 2) Python 实现
# ============================================================
def _feature_fusion_weighted_sum_impl(
    features: torch.Tensor,
    weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    具体实现：
        features: (K, ...)
        weights : (K,) 或 None
    """
    if features.dim() < 1:
        raise ValueError("features must have at least 1 dimension, with shape (K, ...)")

    K = features.shape[0]

    if K == 0:
        raise ValueError("features first dimension K must be > 0")

    if weights is None:
        weights = torch.ones(
            (K,),
            dtype=features.dtype,
            device=features.device
        )
    else:
        if weights.dim() != 1:
            raise ValueError("weights must have shape (K,)")
        if weights.shape[0] != K:
            raise ValueError("weights length must match features.shape[0]")
        weights = weights.to(device=features.device, dtype=features.dtype)

    # 把 weights reshape 成可广播形式: (K, 1, 1, ..., 1)
    view_shape = (K,) + (1,) * (features.dim() - 1)
    weights_broadcast = weights.view(view_shape)

    fused = (features * weights_broadcast).sum(dim=0)
    return fused


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("feature_fusion_weighted_sum", _feature_fusion_weighted_sum_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _feature_fusion_weighted_sum_meta(
    features: torch.Tensor,
    weights: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.empty(
        features.shape[1:],
        device="meta",
        dtype=features.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("feature_fusion_weighted_sum", _feature_fusion_weighted_sum_meta)


# ============================================================
# 4) Python 包装函数
# ============================================================
def feature_fusion_weighted_sum(
    features: torch.Tensor,
    weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.feature_fusion_weighted_sum

    Args:
        features: (K, ...)
        weights:  (K,) 或 None

    Returns:
        fused: (...)
    """
    return torch.ops.fusionrepo.feature_fusion_weighted_sum(
        features,
        weights
    )