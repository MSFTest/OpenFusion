# -*- coding: utf-8 -*-
"""
pixel_to_grid_normalized

说明：
    该文件实现 “像素坐标 -> grid_sample 归一化坐标” 通用原语，
    供 operator_registry.py 中注册后的
    torch.ops.fusionrepo.pixel_to_grid_normalized 调用。

功能：
    将图像平面上的像素坐标 (u, v) 转换为 grid_sample 所需的
    [-1, 1] 归一化采样坐标。

    该算子只负责坐标变换，不负责：
        - 图像特征采样
        - batch 选择
        - 特征融合
        - 线性映射

适用场景：
    - LoGoNet LocalFusionLayer 中 grid_sample 前的坐标归一化
    - EPNet / EPNet++ 图像点采样前的坐标归一化
    - AVOD / 3D-CVF 等基于 grid_sample 的特征采样前处理

输入：
    pixel_coords: (..., 2)
        像素坐标，最后一维为 (u, v)

    height: int
        特征图高度 H

    width: int
        特征图宽度 W

输出：
    normalized_grid: (..., 2)
        归一化后的采样坐标，最后一维为 (x_norm, y_norm)
        坐标范围理论上为 [-1, 1]
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 1) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _pixel_to_grid_normalized_impl(
    pixel_coords: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    if pixel_coords.shape[-1] != 2:
        raise ValueError("pixel_coords last dim must be 2")

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    dtype = pixel_coords.dtype
    device = pixel_coords.device

    u = pixel_coords[..., 0]
    v = pixel_coords[..., 1]

    if width == 1:
        norm_u = torch.zeros_like(u, device=device, dtype=dtype)
    else:
        norm_u = u / (float(width) - 1.0) * 2.0 - 1.0

    if height == 1:
        norm_v = torch.zeros_like(v, device=device, dtype=dtype)
    else:
        norm_v = v / (float(height) - 1.0) * 2.0 - 1.0

    normalized_grid = torch.stack([norm_u, norm_v], dim=-1)
    return normalized_grid


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pixel_to_grid_normalized", _pixel_to_grid_normalized_impl)


# ============================================================
# 2) Meta 实现
# ============================================================
def _pixel_to_grid_normalized_meta(
    pixel_coords: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    return torch.empty(
        pixel_coords.shape,
        device="meta",
        dtype=pixel_coords.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pixel_to_grid_normalized", _pixel_to_grid_normalized_meta)


# ============================================================
# 3) Python 包装
# ============================================================
def pixel_to_grid_normalized(
    pixel_coords: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    return torch.ops.fusionrepo.pixel_to_grid_normalized(
        pixel_coords,
        int(height),
        int(width)
    )