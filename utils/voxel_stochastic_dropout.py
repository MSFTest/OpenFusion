# -*- coding: utf-8 -*-
"""
voxel_stochastic_dropout

说明：
    该文件实现"体素随机丢弃" VirConv 私有 wrapper 算子。

功能：
    根据距离 bin 对体素进行随机丢弃，实现 STVD（Spatial-Temporal Voxel Dropout）策略。
    - 计算每个体素的距离 bin
    - 根据 discard_rate 随机丢弃低质量体素
    - 返回保留下来的体素坐标和特征

    该算子只负责：
    - 距离 bin 计算
    - 随机丢弃

    不负责：
    - 特征聚合
    - 坐标变换

输入：
    voxel_coords: (M, 3)
        体素坐标索引 (ix, iy, iz)。

    voxel_features: (M, C)
        体素特征。

    bin_ids: (M,)
        每个体素对应的距离 bin id。

    discard_rate: float
        丢弃比例。

    training: bool
        是否训练阶段。

输出：
    voxel_coords_kept: (K, 3)
        保留下来的体素坐标。

    voxel_features_kept: (K, C)
        保留下来的体素特征。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _voxel_stochastic_dropout_impl(
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor,
    bin_ids: torch.Tensor,
    discard_rate: float,
    training: bool
):
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
        raise ValueError(f"voxel_coords must be (M,3), got {tuple(voxel_coords.shape)}")

    if voxel_features.ndim != 2:
        raise ValueError(f"voxel_features must be (M,C), got {tuple(voxel_features.shape)}")

    if voxel_coords.shape[0] != voxel_features.shape[0]:
        raise ValueError("voxel_coords and voxel_features must have same first dimension")

    M = voxel_coords.shape[0]

    if M == 0:
        return voxel_coords.clone(), voxel_features.clone()

    if not training or discard_rate <= 0.0:
        return voxel_coords.clone(), voxel_features.clone()

    if discard_rate >= 1.0:
        # Return empty
        return (
            torch.empty((0, 3), device=voxel_coords.device, dtype=voxel_coords.dtype),
            torch.empty((0, voxel_features.shape[1]), device=voxel_features.device, dtype=voxel_features.dtype)
        )

    # Compute dropout mask based on bin_ids
    # Use bin_ids to determine which voxels to keep
    # For VirConv STVD: discard voxels in certain distance bins with probability discard_rate
    if bin_ids is not None and bin_ids.numel() == M:
        # Compute keep probability based on bin
        # Farther bins have lower quality, so they are more likely to be discarded
        num_bins = bin_ids.max().item() + 1 if M > 0 else 1
        bin_probs = torch.rand(M, device=voxel_coords.device)

        # Bin 0 (closest) has lowest dropout, higher bins have higher dropout
        # Scale dropout by bin: bins closer have lower discard probability
        bin_weight = (bin_ids.float() / max(num_bins - 1, 1)).to(voxel_coords.device)
        threshold = discard_rate * (1.0 - bin_weight * 0.5)  # Reduce discard for far bins

        keep_mask = bin_probs > threshold
    else:
        # Simple random dropout
        keep_mask = torch.rand(M, device=voxel_coords.device) > discard_rate

    # Apply mask
    voxel_coords_kept = voxel_coords[keep_mask]
    voxel_features_kept = voxel_features[keep_mask]

    return voxel_coords_kept.clone(), voxel_features_kept.clone()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("voxel_stochastic_dropout", _voxel_stochastic_dropout_impl)


# ============================================================
# Meta
# ============================================================

def _voxel_stochastic_dropout_meta(
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor,
    bin_ids: torch.Tensor,
    discard_rate: float,
    training: bool
):
    m = voxel_coords.shape[0]
    return (
        torch.empty((m, 3), device="meta", dtype=voxel_coords.dtype),
        torch.empty((m, voxel_features.shape[1]), device="meta", dtype=voxel_features.dtype)
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("voxel_stochastic_dropout", _voxel_stochastic_dropout_meta)


# ============================================================
# Python wrapper
# ============================================================

def voxel_stochastic_dropout(
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor,
    bin_ids: torch.Tensor,
    discard_rate: float,
    training: bool
):
    """
    Python wrapper：内部调用 torch.ops.fusionrepo.voxel_stochastic_dropout

    Args:
        voxel_coords: (M, 3)
        voxel_features: (M, C)
        bin_ids: (M,)
        discard_rate: float
        training: bool

    Returns:
        voxel_coords_kept: (K, 3)
        voxel_features_kept: (K, C)
    """
    return torch.ops.fusionrepo.voxel_stochastic_dropout(
        voxel_coords,
        voxel_features,
        bin_ids,
        discard_rate,
        training
    )