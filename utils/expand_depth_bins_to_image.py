# -*- coding: utf-8 -*-
"""
expand_depth_bins_to_image

说明：
    该文件将“depth bins 扩展到图像空间”注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.expand_depth_bins_to_image(depth_bins, height, width)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将 1D depth bins 扩展到图像空间，生成每个像素共享的一组深度离散表示。

输入：
    - depth_bins: (num_bins,)
    - height: int
    - width: int

输出：
    - depth_map_bins: (height, width, num_bins)

说明：
    该算子只做“view + expand”形式的图像级展开，
    不改变 depth bins 数值本身，也不包含其他任务相关后处理。

Args:
    depth_bins: (num_bins,)
        一维深度 bins 张量。

    height: int
        图像/特征图高度。

    width: int
        图像/特征图宽度。

Returns:
    depth_map_bins: (height, width, num_bins)
        扩展后的图像级深度 bins 表示。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _expand_depth_bins_to_image_impl(
    depth_bins: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    具体实现：
        - depth_bins: (num_bins,)
        - return: (height, width, num_bins)
    """
    if depth_bins.dim() != 1:
        raise ValueError("depth_bins must have shape (num_bins,)")

    if height <= 0:
        raise ValueError("height must be positive")
    if width <= 0:
        raise ValueError("width must be positive")

    num_bins = depth_bins.shape[0]
    depth_map_bins = depth_bins.view(1, 1, num_bins).expand(height, width, num_bins)
    return depth_map_bins


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("expand_depth_bins_to_image", _expand_depth_bins_to_image_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _expand_depth_bins_to_image_meta(
    depth_bins: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    if depth_bins.dim() != 1:
        raise ValueError("depth_bins must have shape (num_bins,)")

    if height <= 0:
        height = 1
    if width <= 0:
        width = 1

    num_bins = max(int(depth_bins.shape[0]), 1)

    return torch.empty(
        (height, width, num_bins),
        device="meta",
        dtype=depth_bins.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("expand_depth_bins_to_image", _expand_depth_bins_to_image_meta)


# ============================================================
# 4) 可选 Python 包装
# ============================================================
def expand_depth_bins_to_image(
    depth_bins: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.expand_depth_bins_to_image
    """
    return torch.ops.fusionrepo.expand_depth_bins_to_image(
        depth_bins, int(height), int(width)
    )