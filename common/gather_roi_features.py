# -*- coding: utf-8 -*-
"""
gather_roi_features

说明：
    该文件实现 “ROI内特征收集” 前置原语，
    供 operator_registry.py 中注册后的
    torch.ops.fusionrepo.gather_roi_features 调用。

功能：
    根据 ROI 范围，从 voxel / point 特征中收集每个 ROI 内部的局部特征。

    该算子只负责：
        1. 判断哪些点 / voxel 落在 ROI 内
        2. 将这些特征整理成固定张量
        3. 统计每个 ROI 的有效特征数量

    不负责：
        - mean / max / sum 聚合
        - 后续分类检测
        - 特征变换

输入：
    rois: (K, 7)
        ROI 参数，格式为：
        (x, y, z, dx, dy, dz, yaw)

    voxel_coords: (N, 3)
        每个 voxel / point 的三维坐标
        格式为：
        (x, y, z)

    voxel_features: (N, C)
        每个 voxel / point 对应的特征

输出：
    gathered_features: (K, N, C)
        每个 ROI 内收集到的特征，未命中的位置补 0

    valid_counts: (K,)
        每个 ROI 内有效特征数量
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 1) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _gather_roi_features_impl(
    rois: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor
):
    if rois.dim() != 2 or rois.shape[1] != 7:
        raise ValueError("rois must have shape (K, 7)")

    if voxel_coords.dim() != 2 or voxel_coords.shape[1] != 3:
        raise ValueError("voxel_coords must have shape (N, 3)")

    if voxel_features.dim() != 2:
        raise ValueError("voxel_features must have shape (N, C)")

    if voxel_coords.shape[0] != voxel_features.shape[0]:
        raise ValueError("voxel_coords and voxel_features must have the same first dim")

    device = voxel_features.device
    dtype = voxel_features.dtype

    rois = rois.to(device=device, dtype=dtype)
    voxel_coords = voxel_coords.to(device=device, dtype=dtype)

    K = rois.shape[0]
    N = voxel_coords.shape[0]
    C = voxel_features.shape[1]

    gathered_features = torch.zeros((K, N, C), device=device, dtype=dtype)
    valid_counts = torch.zeros((K,), device=device, dtype=torch.long)

    for i in range(K):
        cx, cy, cz, dx, dy, dz, yaw = rois[i]

        rel = voxel_coords - rois[i, 0:3].unsqueeze(0)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        local_x = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        local_y = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw
        local_z = rel[:, 2]

        mask = (
            (local_x >= -dx / 2) & (local_x <= dx / 2) &
            (local_y >= -dy / 2) & (local_y <= dy / 2) &
            (local_z >= -dz / 2) & (local_z <= dz / 2)
        )

        count = int(mask.sum().item())
        valid_counts[i] = count

        if count > 0:
            gathered_features[i, mask, :] = voxel_features[mask]

    return gathered_features, valid_counts


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("gather_roi_features", _gather_roi_features_impl)


# ============================================================
# 2) Meta 实现
# ============================================================
def _gather_roi_features_meta(
    rois: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor
):
    K = rois.shape[0]
    N = voxel_coords.shape[0]
    C = voxel_features.shape[1]

    gathered_features = torch.empty((K, N, C), device="meta", dtype=voxel_features.dtype)
    valid_counts = torch.empty((K,), device="meta", dtype=torch.long)

    return gathered_features, valid_counts


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("gather_roi_features", _gather_roi_features_meta)


# ============================================================
# 3) Python 包装
# ============================================================
def gather_roi_features(
    rois: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor
):
    return torch.ops.fusionrepo.gather_roi_features(
        rois,
        voxel_coords,
        voxel_features
    )
