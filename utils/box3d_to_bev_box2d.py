# -*- coding: utf-8 -*-
"""
box3d_to_bev_box2d

说明：
    将 3D 检测框投影到 BEV 平面生成二维 BEV 框。

注册算子：
    torch.ops.fusionrepo.box3d_to_bev_box2d(boxes_3d, bev_extents)

功能：
    将输入的 3D box：
        [x, y, z, dx, dy, dz]

    投影为 BEV 平面框：
        [x1, z1, x2, z2]

    并归一化到 [0,1]。

等价替代：
    avod::anchor_to_bev_projection
    avod::proposal_to_bev_projection
"""

from __future__ import annotations
import torch
from torch.library import Library




# ============================================================
# 2) Implementation
# ============================================================

def _box3d_to_bev_box2d_impl(
        boxes_3d: torch.Tensor,
        bev_extents: torch.Tensor):

    x = boxes_3d[:, 0]
    z = boxes_3d[:, 2]

    dx = boxes_3d[:, 3] * 0.5
    dz = boxes_3d[:, 5] * 0.5

    # BEV corners
    x1 = x - dx
    x2 = x + dx
    z1 = z - dz
    z2 = z + dz

    x_min = bev_extents[0, 0]
    x_max = bev_extents[0, 1]

    z_min = bev_extents[1, 0]
    z_max = bev_extents[1, 1]

    # 坐标系转换
    bev_x1 = x1 - x_min
    bev_x2 = x2 - x_min

    bev_z1 = z_max - z2
    bev_z2 = z_max - z1

    bev_width = x_max - x_min
    bev_height = z_max - z_min

    boxes_norm = torch.stack(
        [
            bev_x1 / bev_width,
            bev_z1 / bev_height,
            bev_x2 / bev_width,
            bev_z2 / bev_height
        ],
        dim=1
    )

    return boxes_norm


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "box3d_to_bev_box2d",
    _box3d_to_bev_box2d_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================

def _box3d_to_bev_box2d_meta(
        boxes_3d: torch.Tensor,
        bev_extents: torch.Tensor):

    return torch.empty(
        boxes_3d.shape[0],
        4,
        device="meta",
        dtype=boxes_3d.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "box3d_to_bev_box2d",
    _box3d_to_bev_box2d_meta
)


# ============================================================
# 4) Python wrapper（可选）
# ============================================================

def box3d_to_bev_box2d(
        boxes_3d: torch.Tensor,
        bev_extents: torch.Tensor):

    return torch.ops.fusionrepo.box3d_to_bev_box2d(
        boxes_3d,
        bev_extents
    )