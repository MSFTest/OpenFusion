# -*- coding: utf-8 -*-
"""
fuse_depth_slice_samples

说明：
    该文件将“两个深度切片采样结果的深度方向融合”注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.fuse_depth_slice_samples(
            sampled0,
            sampled1,
            depth_alpha
        )

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将两个相邻深度切片上的 2D 双线性采样结果，
    按照深度方向插值权重 depth_alpha 做线性融合，
    恢复三线性插值在深度维度上的那一部分。

输入：
    sampled0     : (N, C, 1) 或 (N, C)
    sampled1     : (N, C, 1) 或 (N, C)
    depth_alpha  : (N, 1)

输出：
    fused        : (N, C)
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现
# ============================================================
def _fuse_depth_slice_samples_impl(
    sampled0: torch.Tensor,
    sampled1: torch.Tensor,
    depth_alpha: torch.Tensor
) -> torch.Tensor:
    """
    具体实现：
        sampled0    : (N, C, 1) 或 (N, C)
        sampled1    : (N, C, 1) 或 (N, C)
        depth_alpha : (N, 1)

    返回：
        fused       : (N, C)
    """
    if sampled0.shape != sampled1.shape:
        raise ValueError("sampled0 and sampled1 must have the same shape")

    if depth_alpha.dim() != 2 or depth_alpha.shape[1] != 1:
        raise ValueError("depth_alpha must have shape (N, 1)")

    x0 = sampled0
    x1 = sampled1

    if x0.dim() == 3:
        if x0.shape[-1] != 1:
            raise ValueError("when sampled is 3D, last dim must be 1")
        x0 = x0.squeeze(-1)  # (N, C)
        x1 = x1.squeeze(-1)  # (N, C)

    if x0.dim() != 2:
        raise ValueError("sampled tensors must have shape (N, C, 1) or (N, C)")

    if x0.shape[0] != depth_alpha.shape[0]:
        raise ValueError("depth_alpha.shape[0] must equal sampled.shape[0]")

    alpha = depth_alpha.to(device=x0.device, dtype=x0.dtype)  # (N,1)

    fused = (1.0 - alpha) * x0 + alpha * x1
    return fused.contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "fuse_depth_slice_samples",
    _fuse_depth_slice_samples_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================
def _fuse_depth_slice_samples_meta(
    sampled0: torch.Tensor,
    sampled1: torch.Tensor,
    depth_alpha: torch.Tensor
) -> torch.Tensor:
    if sampled0.dim() == 3:
        return torch.empty((sampled0.shape[0], sampled0.shape[1]), device="meta", dtype=sampled0.dtype)

    return torch.empty((sampled0.shape[0], sampled0.shape[1]), device="meta", dtype=sampled0.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "fuse_depth_slice_samples",
    _fuse_depth_slice_samples_meta
)


# ============================================================
# 4) Python 包装
# ============================================================
def fuse_depth_slice_samples(
    sampled0: torch.Tensor,
    sampled1: torch.Tensor,
    depth_alpha: torch.Tensor
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.fuse_depth_slice_samples
    """
    return torch.ops.fusionrepo.fuse_depth_slice_samples(
        sampled0,
        sampled1,
        depth_alpha
    )