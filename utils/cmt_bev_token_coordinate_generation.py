# -*- coding: utf-8 -*-
"""CMT LiDAR-BEV token 网格中心坐标生成算子。"""

from __future__ import annotations

import torch
from torch.library import Library

from utils.bevdet_compute_query_position import compute_query_position


def _impl(bev_feature: torch.Tensor) -> torch.Tensor:
    """生成 CMT BEV token 的归一化网格中心坐标。"""
    if bev_feature.ndim != 4:
        raise ValueError("bev_feature must have shape [B,C,H,W]")
    height, width = bev_feature.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("BEV spatial dimensions must be positive")
    indices = torch.arange(height * width, device=bev_feature.device, dtype=torch.long)
    centers = compute_query_position(indices, height, width)
    scale = centers.new_tensor((width, height))
    return centers / scale


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("cmt_bev_token_coordinates", _impl)


def _meta(bev_feature: torch.Tensor) -> torch.Tensor:
    if bev_feature.ndim != 4:
        raise ValueError("bev_feature must have shape [B,C,H,W]")
    if bev_feature.shape[-2] <= 0 or bev_feature.shape[-1] <= 0:
        raise ValueError("BEV spatial dimensions must be positive")
    return torch.empty(
        (bev_feature.shape[-2] * bev_feature.shape[-1], 2),
        device="meta",
        dtype=torch.float32,
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("cmt_bev_token_coordinates", _meta)


def cmt_bev_token_coordinates(bev_feature: torch.Tensor) -> torch.Tensor:
    """生成 CMT BEV token 中心坐标。

    Args:
        bev_feature: ``[B,C,H,W]``，任意数值 dtype，当前 LiDAR-BEV 特征；
            仅使用 shape 与 device，要求 ``H,W>0``。

    Returns:
        ``float32[H*W,2]``，与输入同 device、行优先 contiguous 布局；每项为
        当前 BEV 平面归一化 ``(x,y)`` 中心，采用半格偏移，范围严格位于
        ``(0,1)``。该离散网格构造不对输入特征值求导。
    """
    return torch.ops.fusionrepo.cmt_bev_token_coordinates(bev_feature)
