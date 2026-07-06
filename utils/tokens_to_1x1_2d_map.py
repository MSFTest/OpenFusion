# -*- coding: utf-8 -*-
"""
tokens_to_1x1_2d_map

说明：
    该文件将“token / voxel 特征 -> 1x1 Conv 可处理的二维特征图形式”
    注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.tokens_to_1x1_2d_map(x)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将形如 (N, C) 或 (B, N, C) 的 token / voxel 特征，
    转换为 channel_reduce_1x1_2d 可处理的 (B, C, H, W) 形式。

形状约定：
    - (N, C)     -> (1, C, N, 1)
    - (B, N, C)  -> (B, C, N, 1)

说明：
    该算子仅做形状重排，不改变数值语义，不引入可学习参数。
    主要用于把 HMFI 中的 voxel/token 特征适配到
    channel_reduce_1x1_2d 的输入接口。

Args:
    x: (N, C) 或 (B, N, C)
        token / voxel 特征张量

Returns:
    x_map: (1, C, N, 1) 或 (B, C, N, 1)
        适配后的二维特征图表示
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现
# ============================================================
def _tokens_to_1x1_2d_map_impl(
    x: torch.Tensor
) -> torch.Tensor:
    """
    具体实现：
        - (N, C)    -> (1, C, N, 1)
        - (B, N, C) -> (B, C, N, 1)
    """
    if x.dim() == 2:
        # (N, C) -> (1, N, C) -> (1, C, N, 1)
        return x.unsqueeze(0).transpose(1, 2).unsqueeze(-1)

    if x.dim() == 3:
        # (B, N, C) -> (B, C, N, 1)
        return x.transpose(1, 2).unsqueeze(-1)

    raise ValueError("x must have shape (N, C) or (B, N, C)")


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("tokens_to_1x1_2d_map", _tokens_to_1x1_2d_map_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _tokens_to_1x1_2d_map_meta(
    x: torch.Tensor
) -> torch.Tensor:
    if x.dim() == 2:
        n, c = x.shape
        return torch.empty((1, c, n, 1), device="meta", dtype=x.dtype)

    if x.dim() == 3:
        b, n, c = x.shape
        return torch.empty((b, c, n, 1), device="meta", dtype=x.dtype)

    raise ValueError("x must have shape (N, C) or (B, N, C)")


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("tokens_to_1x1_2d_map", _tokens_to_1x1_2d_map_meta)


# ============================================================
# 4) Python 包装
# ============================================================
def tokens_to_1x1_2d_map(
    x: torch.Tensor
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.tokens_to_1x1_2d_map
    """
    return torch.ops.fusionrepo.tokens_to_1x1_2d_map(x)