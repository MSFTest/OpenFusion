# -*- coding: utf-8 -*-
"""
projected_pixel_valid_mask

说明：
    投影像素有效性判断通用算子，注册为：
        torch.ops.fusionrepo.projected_pixel_valid_mask(uv, depth, height, width, eps)

    该算子用于 LiDAR / voxel / RoI 网格点投影到图像平面后，判断投影点
    是否位于图像范围内且具有有效正深度。坐标系约定为图像像素坐标系：
    uv[..., 0] 为水平像素 u，uv[..., 1] 为垂直像素 v。

Args:
    uv: Tensor
        图像像素坐标，shape 为 (..., 2)。
    depth: Tensor
        投影深度，shape 为 (...) 或 (..., 1)。
    height: int
        图像高度 H。
    width: int
        图像宽度 W。
    eps: float
        深度有效阈值，depth > eps 视为有效。

Returns:
    Tensor
        bool 掩码，shape 为 uv.shape[:-1]。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _projected_pixel_valid_mask_impl(
    uv: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
    eps: float = 0.0,
) -> torch.Tensor:
    if uv.shape[-1] != 2:
        raise ValueError("uv last dim must be 2")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    d = depth[..., 0] if depth.dim() > 0 and depth.shape[-1] == 1 else depth
    if d.shape != uv.shape[:-1]:
        raise ValueError("depth shape must be uv.shape[:-1] or uv.shape[:-1] + (1,)")

    u = uv[..., 0]
    v = uv[..., 1]
    return (
        (d > float(eps))
        & (u >= 0)
        & (u <= float(width - 1))
        & (v >= 0)
        & (v <= float(height - 1))
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("projected_pixel_valid_mask", _projected_pixel_valid_mask_impl)


def _projected_pixel_valid_mask_meta(
    uv: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
    eps: float = 0.0,
) -> torch.Tensor:
    return torch.empty(uv.shape[:-1], device="meta", dtype=torch.bool)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("projected_pixel_valid_mask", _projected_pixel_valid_mask_meta)


def projected_pixel_valid_mask(
    uv: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
    eps: float = 0.0,
) -> torch.Tensor:
    return torch.ops.fusionrepo.projected_pixel_valid_mask(
        uv,
        depth,
        int(height),
        int(width),
        float(eps),
    )
