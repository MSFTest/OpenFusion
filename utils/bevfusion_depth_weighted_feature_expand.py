# -*- coding: utf-8 -*-
"""
BEVfusion 深度加权图像特征展开 wrapper。

该算子保留 BEVfusion 原始输入输出语义，内部复用通用
image_frustum_feature_construction 原语完成图像特征与深度分布的外积展开。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _bevfusion_depth_weighted_feature_expand_impl(
    img_feats: torch.Tensor,
    depth_prob: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        img_feats: (B, C, H, W)
            相机二维特征图。
        depth_prob: (B, D, H, W)
            每个像素在 D 个深度 bin 上的概率分布。

    Returns:
        feat_3d: (B, D, C, H, W)
            深度概率加权后的 frustum 三维特征。
    """
    if img_feats.dim() != 4:
        raise ValueError("img_feats must have shape (B, C, H, W)")
    if depth_prob.dim() != 4:
        raise ValueError("depth_prob must have shape (B, D, H, W)")
    if img_feats.shape[0] != depth_prob.shape[0]:
        raise ValueError("img_feats and depth_prob must share batch size")
    if img_feats.shape[2:] != depth_prob.shape[2:]:
        raise ValueError("img_feats and depth_prob must share spatial size")

    outputs = []
    for image_feature, depth_dist in zip(img_feats, depth_prob):
        frustum = torch.ops.fusionrepo.image_frustum_feature_construction(
            image_feature.permute(1, 2, 0).contiguous(),
            depth_dist.permute(1, 2, 0).contiguous(),
        )
        outputs.append(frustum.permute(2, 3, 0, 1).contiguous())
    return torch.stack(outputs, dim=0)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "bevfusion_depth_weighted_feature_expand",
    _bevfusion_depth_weighted_feature_expand_impl,
)


def _bevfusion_depth_weighted_feature_expand_meta(
    img_feats: torch.Tensor,
    depth_prob: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (
            img_feats.shape[0],
            depth_prob.shape[1],
            img_feats.shape[1],
            img_feats.shape[2],
            img_feats.shape[3],
        ),
        device="meta",
        dtype=img_feats.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "bevfusion_depth_weighted_feature_expand",
    _bevfusion_depth_weighted_feature_expand_meta,
)


def bevfusion_depth_weighted_feature_expand(
    img_feats: torch.Tensor,
    depth_prob: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.bevfusion_depth_weighted_feature_expand(
        img_feats,
        depth_prob,
    )
