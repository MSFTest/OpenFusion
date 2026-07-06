# -*- coding: utf-8 -*-
"""
feature_gather_by_pixel_index

说明：
    按图像像素坐标从二维特征图、语义图或掩码图中读取对应位置的特征。
    该算子使用离散像素索引，不执行双线性插值；适用于 PointPainting
    点级语义查询、实例掩码像素查询等场景。

Args:
    pixel_coords:
        像素坐标，shape 为 (N, 2)，最后一维为 (u, v)。

    feature_map:
        被查询的二维特征图，支持：
        - (H, W)
        - (H, W, C)
        - (C, H, W)

    rounding_mode:
        坐标转整数方式，支持 "long"、"floor"、"round"。

    layout:
        feature_map 布局，支持 "HW"、"HWC"、"CHW"。

    clamp_coords:
        是否将越界坐标截断到图像范围内。

Returns:
    gathered:
        若 feature_map 为 (H, W)，输出 shape 为 (N,)；
        若 feature_map 为 (H, W, C) 或 (C, H, W)，输出 shape 为 (N, C)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _feature_gather_by_pixel_index_impl(
    pixel_coords: torch.Tensor,
    feature_map: torch.Tensor,
    rounding_mode: str = "long",
    layout: str = "HWC",
    clamp_coords: bool = True,
) -> torch.Tensor:
    if pixel_coords.dim() != 2 or pixel_coords.shape[1] != 2:
        raise ValueError("pixel_coords must have shape (N, 2)")

    layout = str(layout).upper()
    if layout not in ("HW", "HWC", "CHW"):
        raise ValueError("layout must be one of 'HW', 'HWC', or 'CHW'")

    if layout == "HW":
        if feature_map.dim() != 2:
            raise ValueError("feature_map must have shape (H, W) when layout='HW'")
        height, width = feature_map.shape
    elif layout == "HWC":
        if feature_map.dim() != 3:
            raise ValueError("feature_map must have shape (H, W, C) when layout='HWC'")
        height, width = feature_map.shape[:2]
    else:
        if feature_map.dim() != 3:
            raise ValueError("feature_map must have shape (C, H, W) when layout='CHW'")
        height, width = feature_map.shape[-2:]

    mode = str(rounding_mode).lower()
    if mode == "long":
        coords = pixel_coords.long()
    elif mode == "floor":
        coords = torch.floor(pixel_coords).long()
    elif mode == "round":
        coords = torch.round(pixel_coords).long()
    else:
        raise ValueError("rounding_mode must be one of 'long', 'floor', or 'round'")

    u = coords[:, 0]
    v = coords[:, 1]
    if bool(clamp_coords):
        u = u.clamp(min=0, max=int(width) - 1)
        v = v.clamp(min=0, max=int(height) - 1)

    if layout == "HW":
        return feature_map[v, u].contiguous()
    if layout == "HWC":
        return feature_map[v, u, :].contiguous()
    return feature_map[:, v, u].transpose(0, 1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("feature_gather_by_pixel_index", _feature_gather_by_pixel_index_impl)


def _feature_gather_by_pixel_index_meta(
    pixel_coords: torch.Tensor,
    feature_map: torch.Tensor,
    rounding_mode: str = "long",
    layout: str = "HWC",
    clamp_coords: bool = True,
) -> torch.Tensor:
    n = pixel_coords.shape[0]
    layout = str(layout).upper()
    if layout == "HW":
        out_shape = (n,)
    elif layout == "HWC":
        out_shape = (n, feature_map.shape[-1])
    elif layout == "CHW":
        out_shape = (n, feature_map.shape[0])
    else:
        raise ValueError("layout must be one of 'HW', 'HWC', or 'CHW'")
    return torch.empty(out_shape, device="meta", dtype=feature_map.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("feature_gather_by_pixel_index", _feature_gather_by_pixel_index_meta)


def feature_gather_by_pixel_index(
    pixel_coords: torch.Tensor,
    feature_map: torch.Tensor,
    rounding_mode: str = "long",
    layout: str = "HWC",
    clamp_coords: bool = True,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_gather_by_pixel_index(
        pixel_coords,
        feature_map,
        str(rounding_mode),
        str(layout),
        bool(clamp_coords),
    )
