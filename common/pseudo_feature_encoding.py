# -*- coding: utf-8 -*-
"""
pseudo_feature_encoding

说明：
    该文件实现“伪点云 ROI 特征编码”通用函数算子，
    供 operator_registry.py 中注册后的

        torch.ops.fusionrepo.pseudo_feature_encoding

    调用。

功能：
    对裁剪后的伪点云 ROI 点集执行特征编码，
    将不规则点级表示转换为 ROI 级伪点云特征。

    当前实现流程为：
        1. 根据 roi_indices 按 ROI 分组
        2. 可选保留 xyz
        3. 拼接其余属性特征
        4. 对每个 ROI 内点特征做平均池化

输入：
    roi_pseudo_points : Tensor
        shape 为 (N, C)
        每个点格式为：
        [x, y, z, f1, f2, ...]

    roi_indices : Tensor
        shape 为 (N,)
        每个点所属 ROI 的索引

    use_xyz : bool
        是否在编码结果中保留 xyz 坐标

输出：
    roi_pseudo_features : Tensor
        shape 为 (R, C_out)
        R 为 ROI 数量
        C_out 为编码后特征维度
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _pseudo_feature_encoding_impl(
        roi_pseudo_points: torch.Tensor,
        roi_indices: torch.Tensor,
        use_xyz: bool = True) -> torch.Tensor:
    if roi_pseudo_points.dim() != 2 or roi_pseudo_points.size(1) < 3:
        raise ValueError("roi_pseudo_points must have shape (N, C) with C >= 3")

    if roi_indices.dim() != 1 or roi_indices.size(0) != roi_pseudo_points.size(0):
        raise ValueError("roi_indices must have shape (N,)")

    device = roi_pseudo_points.device
    dtype = roi_pseudo_points.dtype

    num_rois = int(roi_indices.max().item()) + 1 if roi_indices.numel() > 0 else 0

    if use_xyz:
        out_dim = roi_pseudo_points.size(1)
    else:
        out_dim = max(roi_pseudo_points.size(1) - 3, 0)

    if num_rois == 0:
        return torch.empty((0, out_dim), device=device, dtype=dtype)

    roi_features = []

    for roi_id in range(num_rois):
        mask = roi_indices == roi_id
        pts = roi_pseudo_points[mask]

        if pts.numel() == 0:
            roi_features.append(
                torch.zeros((out_dim,), device=device, dtype=dtype)
            )
            continue

        features = []

        if use_xyz:
            features.append(pts[:, :3])

        if pts.size(1) > 3:
            features.append(pts[:, 3:])

        if len(features) == 0:
            roi_feat = torch.zeros((out_dim,), device=device, dtype=dtype)
        else:
            roi_feat = torch.cat(features, dim=1)
            roi_feat = roi_feat.mean(dim=0)

        roi_features.append(roi_feat)

    roi_pseudo_features = torch.stack(roi_features, dim=0)
    return roi_pseudo_features


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "pseudo_feature_encoding",
    _pseudo_feature_encoding_impl
)


# ============================================================
# Meta
# ============================================================

def _pseudo_feature_encoding_meta(
        roi_pseudo_points: torch.Tensor,
        roi_indices: torch.Tensor,
        use_xyz: bool = True) -> torch.Tensor:
    if roi_pseudo_points.dim() != 2 or roi_pseudo_points.size(1) < 3:
        raise ValueError("roi_pseudo_points must have shape (N, C) with C >= 3")

    num_rois = 0

    if use_xyz:
        out_dim = roi_pseudo_points.size(1)
    else:
        out_dim = max(roi_pseudo_points.size(1) - 3, 0)

    return torch.empty(
        (num_rois, out_dim),
        device="meta",
        dtype=roi_pseudo_points.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "pseudo_feature_encoding",
    _pseudo_feature_encoding_meta
)


# ============================================================
# wrapper
# ============================================================

def pseudo_feature_encoding(
        roi_pseudo_points: torch.Tensor,
        roi_indices: torch.Tensor,
        use_xyz: bool = True) -> torch.Tensor:
    return torch.ops.fusionrepo.pseudo_feature_encoding(
        roi_pseudo_points,
        roi_indices,
        bool(use_xyz)
    )