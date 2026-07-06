# -*- coding: utf-8 -*-
"""
feature_sampling_by_grid

说明：
    统一实现跨模型的特征采样算子。

注册算子：
    torch.ops.fusionrepo.feature_sampling_by_grid(feature_map, grid)

功能：
    根据连续采样坐标 grid，从特征图 feature_map 中进行双线性插值采样，
    返回对应位置的特征表示。

等价替代：
    avod::bev_crop_resize_rpn / det
    avod::image_crop_resize_rpn / det
    il_fusion::image_point_sampling
    li_fusion::image_sampler
    cvf::camera_to_bev_feature_projection

输入：
    feature_map : (B , C , H , W)

    grid :
        (B , N , 2)   或
        (B , N , 1 , 2)

        坐标范围 [-1 , 1]

输出：
    sampled_features :
        (B , C , N)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.library import Library





# ============================================================
# 2) Implementation
# ============================================================

def _feature_sampling_by_grid_impl(
        feature_map: torch.Tensor,
        grid: torch.Tensor):

    if feature_map.dim() != 4:
        raise ValueError("feature_map must be (B,C,H,W)")

    if grid.dim() == 3:
        grid = grid.unsqueeze(2)   # (B,N,1,2)

    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        align_corners=True
    )  # (B,C,N,1)

    sampled = sampled.squeeze(-1)  # (B,C,N)

    return sampled.contiguous()


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "feature_sampling_by_grid",
    _feature_sampling_by_grid_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================

def _feature_sampling_by_grid_meta(
        feature_map: torch.Tensor,
        grid: torch.Tensor):

    B, C, _, _ = feature_map.shape

    if grid.dim() == 3:
        N = grid.shape[1]
    else:
        N = grid.shape[1]

    return torch.empty(
        (B, C, N),
        device="meta",
        dtype=feature_map.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "feature_sampling_by_grid",
    _feature_sampling_by_grid_meta
)


# ============================================================
# 4) Python wrapper
# ============================================================

def feature_sampling_by_grid(
        feature_map: torch.Tensor,
        grid: torch.Tensor):

    return torch.ops.fusionrepo.feature_sampling_by_grid(
        feature_map,
        grid
    )