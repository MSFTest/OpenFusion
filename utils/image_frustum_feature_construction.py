# -*- coding: utf-8 -*-
"""
image_frustum_feature_construction

说明：
    该文件将“图像 Frustum 特征构建”注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.image_frustum_feature_construction(
            image_features,
            depth_bins
        )

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    通过将 2D 图像特征与像素级深度离散分布在深度维度上进行外积扩展，
    构建沿视线方向展开的图像 Frustum 特征。

数学形式：
    G_{m,n} = F_{m,n} ⊗ D_{m,n}

输入：
    image_features: (H, W, C)
        2D 图像 Backbone 输出特征图。

    depth_bins: (H, W, R)
        每个像素对应的离散深度 bins / 深度分布。

输出：
    frustum_features: (H, W, R, C)
        沿深度方向展开后的图像 Frustum 特征。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _image_frustum_feature_construction_impl(
    image_features: torch.Tensor,
    depth_bins: torch.Tensor
) -> torch.Tensor:
    """
    具体实现：
        image_features : (H, W, C)
        depth_bins     : (H, W, R)

    返回：
        frustum_features : (H, W, R, C)
    """
    if image_features.dim() != 3:
        raise ValueError("image_features must have shape (H, W, C)")

    if depth_bins.dim() != 3:
        raise ValueError("depth_bins must have shape (H, W, R)")

    h1, w1, _ = image_features.shape
    h2, w2, _ = depth_bins.shape

    if h1 != h2 or w1 != w2:
        raise ValueError("image_features and depth_bins must share the same H and W")

    f = image_features.unsqueeze(2)   # (H, W, 1, C)
    d = depth_bins.unsqueeze(-1)      # (H, W, R, 1)

    frustum_features = f * d          # (H, W, R, C)
    return frustum_features.contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "image_frustum_feature_construction",
    _image_frustum_feature_construction_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================
def _image_frustum_feature_construction_meta(
    image_features: torch.Tensor,
    depth_bins: torch.Tensor
) -> torch.Tensor:
    if image_features.dim() != 3:
        raise ValueError("image_features must have shape (H, W, C)")

    if depth_bins.dim() != 3:
        raise ValueError("depth_bins must have shape (H, W, R)")

    h1, w1, c = image_features.shape
    h2, w2, r = depth_bins.shape

    if h1 != h2 or w1 != w2:
        raise ValueError("image_features and depth_bins must share the same H and W")

    return torch.empty((h1, w1, r, c), device="meta", dtype=image_features.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "image_frustum_feature_construction",
    _image_frustum_feature_construction_meta
)


# ============================================================
# 4) Python 包装
# ============================================================
def image_frustum_feature_construction(
    image_features: torch.Tensor,
    depth_bins: torch.Tensor
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.image_frustum_feature_construction
    """
    return torch.ops.fusionrepo.image_frustum_feature_construction(
        image_features,
        depth_bins
    )