# -*- coding: utf-8 -*-
"""
crop_points_in_3d_boxes

说明：
    该文件实现“3D ROI 内点筛选 / 分组”通用函数算子，
    用于根据 3D ROI 的空间范围，从点集 / 点特征中筛选出各个 ROI 内部的点，
    并组织成按 ROI 分组的稠密张量与有效性掩码。

功能：
    输入：
        - rois: (B, M, 8)
        - points_xyz: (N, 3)
        - points_feat: (N, C)

    输出：
        - grouped_features: (B, M, N, C)
        - grouped_mask:     (B, M, N)

    其中：
        grouped_features[b, m, n, :] 表示第 b 个 batch 下第 m 个 ROI
        对第 n 个点的保留结果；若该点不在 ROI 内，则该位置为 0。
        grouped_mask[b, m, n] 表示该点是否落在对应 ROI 内。

注意：
    该算子只负责：
        1. 判断点是否落在 3D ROI 内
        2. 按 ROI 对点特征进行分组
        3. 输出分组后的特征张量与有效掩码

    该算子不负责：
        - 池化
        - 归约
        - 融合
        - 线性映射
        - 激活函数
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# Implementation
# ============================================================

def _crop_points_in_3d_boxes_impl(
        rois: torch.Tensor,
        points_xyz: torch.Tensor,
        points_feat: torch.Tensor):
    """
    Args:
        rois: (B, M, 8)
            [batch_idx, x, y, z, dx, dy, dz, yaw]
            XY 平面使用 yaw 旋转后的 ROI 局部坐标做筛选。

        points_xyz: (N, 3)
        points_feat: (N, C)

    Returns:
        grouped_features: (B, M, N, C)
        grouped_mask: (B, M, N)
    """
    if rois.dim() != 3 or rois.size(-1) != 8:
        raise ValueError("rois must have shape (B, M, 8)")

    if points_xyz.dim() != 2 or points_xyz.size(-1) != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    if points_feat.dim() != 2:
        raise ValueError("points_feat must have shape (N, C)")

    if points_xyz.size(0) != points_feat.size(0):
        raise ValueError("points_xyz and points_feat must have the same number of points")

    B, M, _ = rois.shape
    N, C = points_feat.shape

    device = points_feat.device
    dtype = points_feat.dtype

    grouped_features = torch.zeros(
        (B, M, N, C),
        device=device,
        dtype=dtype
    )

    grouped_mask = torch.zeros(
        (B, M, N),
        device=device,
        dtype=torch.bool
    )

    px = points_xyz[:, 0]
    py = points_xyz[:, 1]
    pz = points_xyz[:, 2]

    for b in range(B):
        for m in range(M):
            roi = rois[b, m]

            cx, cy, cz = roi[1:4]
            dx, dy, dz, yaw = roi[4:8]

            rel_x = px - cx
            rel_y = py - cy
            # CPU torch does not implement half-precision trigonometry.
            trig_yaw = yaw.float() if yaw.dtype in (torch.float16, torch.bfloat16) else yaw
            cos_yaw = torch.cos(trig_yaw).to(rel_x.dtype)
            sin_yaw = torch.sin(trig_yaw).to(rel_y.dtype)
            local_x = rel_x * cos_yaw + rel_y * sin_yaw
            local_y = -rel_x * sin_yaw + rel_y * cos_yaw
            rel_z = pz - cz

            mask = (
                (local_x.abs() <= dx / 2) &
                (local_y.abs() <= dy / 2) &
                (rel_z.abs() <= dz / 2)
            )

            grouped_mask[b, m] = mask
            grouped_features[b, m] = torch.where(
                mask.unsqueeze(-1), points_feat, torch.zeros_like(points_feat)
            )

    return grouped_features, grouped_mask


_lib_impl = Library(
    "fusionrepo",
    "IMPL",
    "CompositeExplicitAutograd"
)

_lib_impl.impl(
    "crop_points_in_3d_boxes",
    _crop_points_in_3d_boxes_impl
)


# ============================================================
# Meta
# ============================================================

def _crop_points_in_3d_boxes_meta(
        rois: torch.Tensor,
        points_xyz: torch.Tensor,
        points_feat: torch.Tensor):
    """
    Meta function.

    Returns:
        grouped_features: (B, M, N, C)
        grouped_mask: (B, M, N)
    """
    if rois.dim() != 3 or rois.size(-1) != 8:
        raise ValueError("rois must have shape (B, M, 8)")

    if points_xyz.dim() != 2 or points_xyz.size(-1) != 3:
        raise ValueError("points_xyz must have shape (N, 3)")

    if points_feat.dim() != 2:
        raise ValueError("points_feat must have shape (N, C)")

    B, M, _ = rois.shape
    N, C = points_feat.shape

    grouped_features = torch.empty(
        (B, M, N, C),
        device="meta",
        dtype=points_feat.dtype
    )

    grouped_mask = torch.empty(
        (B, M, N),
        device="meta",
        dtype=torch.bool
    )

    return grouped_features, grouped_mask


_lib_meta = Library("fusionrepo", "IMPL", "Meta")

_lib_meta.impl(
    "crop_points_in_3d_boxes",
    _crop_points_in_3d_boxes_meta
)


# ============================================================
# wrapper
# ============================================================

def crop_points_in_3d_boxes(
        rois: torch.Tensor,
        points_xyz: torch.Tensor,
        points_feat: torch.Tensor):
    """
    Returns:
        grouped_features: (B, M, N, C)
        grouped_mask: (B, M, N)
    """
    return torch.ops.fusionrepo.crop_points_in_3d_boxes(
        rois,
        points_xyz,
        points_feat
    )


def iter_crop_points_in_3d_boxes(
        rois: torch.Tensor,
        points_xyz: torch.Tensor,
        points_feat: torch.Tensor,
        roi_chunk_size: int = 2):
    """Yield ``(roi_start, features, mask)`` for bounded ROI slices.

    Cropping is independent for each ROI, so slicing only the M axis preserves
    the operator's values and autograd behavior while bounding the dense
    ``(B, M, N, C)`` temporary to ``(B, roi_chunk_size, N, C)``.
    """
    roi_chunk_size = int(roi_chunk_size)
    if roi_chunk_size <= 0:
        raise ValueError("roi_chunk_size must be a positive integer")

    for roi_start in range(0, rois.shape[1], roi_chunk_size):
        roi_end = min(roi_start + roi_chunk_size, rois.shape[1])
        grouped_features, grouped_mask = crop_points_in_3d_boxes(
            rois[:, roi_start:roi_end], points_xyz, points_feat
        )
        yield roi_start, grouped_features, grouped_mask
