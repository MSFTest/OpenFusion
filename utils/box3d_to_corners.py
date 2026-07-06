# -*- coding: utf-8 -*-
"""
box3d_to_corners

注册算子：
    torch.ops.fusionrepo.box3d_to_corners(boxes_3d)

功能：
    将 AVOD 3D box 转换为 8 个 corner
"""

from __future__ import annotations
import torch
from torch.library import Library




# ============================================================
# 2) Implementation
# ============================================================

def _box3d_to_corners_impl(boxes_3d: torch.Tensor):

    x = boxes_3d[:, 0]
    y = boxes_3d[:, 1]
    z = boxes_3d[:, 2]

    dx = boxes_3d[:, 3] * 0.5
    dy = boxes_3d[:, 4]
    dz = boxes_3d[:, 5] * 0.5

    corners_x = torch.stack(
        [x + dx, x + dx, x - dx, x - dx,
         x + dx, x + dx, x - dx, x - dx],
        dim=1
    )

    corners_y = torch.stack(
        [y, y, y, y,
         y - dy, y - dy, y - dy, y - dy],
        dim=1
    )

    corners_z = torch.stack(
        [z + dz, z - dz, z - dz, z + dz,
         z + dz, z - dz, z - dz, z + dz],
        dim=1
    )

    corners = torch.stack(
        [corners_x, corners_y, corners_z],
        dim=2
    )

    return corners


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("box3d_to_corners", _box3d_to_corners_impl)


# ============================================================
# 3) Meta
# ============================================================

def _box3d_to_corners_meta(boxes_3d: torch.Tensor):

    return torch.empty(
        boxes_3d.shape[0],
        8,
        3,
        device="meta",
        dtype=boxes_3d.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("box3d_to_corners", _box3d_to_corners_meta)