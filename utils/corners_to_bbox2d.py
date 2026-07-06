# -*- coding: utf-8 -*-
"""
corners_to_bbox2d

注册算子：
    torch.ops.fusionrepo.corners_to_bbox2d(corners_pixels, image_shape)

功能：
    将投影后的 corner 像素坐标转换为 2D bbox
"""

from __future__ import annotations
import torch
from torch.library import Library





# ============================================================
# 2) Implementation
# ============================================================

def _corners_to_bbox2d_impl(
        corners_pixels: torch.Tensor,
        image_shape: torch.Tensor):

    proj_x = corners_pixels[..., 0]
    proj_y = corners_pixels[..., 1]

    x1 = proj_x.min(dim=1).values
    y1 = proj_y.min(dim=1).values
    x2 = proj_x.max(dim=1).values
    y2 = proj_y.max(dim=1).values

    H = image_shape[0]
    W = image_shape[1]

    boxes_norm = torch.stack(
        [x1 / W, y1 / H, x2 / W, y2 / H],
        dim=1
    )

    return boxes_norm


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("corners_to_bbox2d", _corners_to_bbox2d_impl)


# ============================================================
# 3) Meta
# ============================================================

def _corners_to_bbox2d_meta(
        corners_pixels: torch.Tensor,
        image_shape: torch.Tensor):

    return torch.empty(
        corners_pixels.shape[0],
        4,
        device="meta",
        dtype=corners_pixels.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("corners_to_bbox2d", _corners_to_bbox2d_meta)