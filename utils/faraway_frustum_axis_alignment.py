# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有 Frustum 坐标轴对齐算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_frustum_axis_alignment_impl(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    frustum_angles: torch.Tensor,
) -> torch.Tensor:
    """按实例绕相机坐标系 Y 轴旋转 Frustum 点云。

    Args:
        clusters: ``(K, P, 3)``，相机坐标系分组点云。
        counts: ``(K,)``，每组有效点数。
        frustum_angles: ``(K,)``，绕 Y 轴旋转角，单位为弧度。

    Returns:
        aligned: ``(K, P, 3)``，旋转后的点云，padding 位置为零。
    """
    if clusters.dim() != 3 or clusters.shape[-1] != 3:
        raise ValueError("clusters must have shape (K, P, 3)")
    if counts.shape != (clusters.shape[0],):
        raise ValueError("counts must have shape (K,)")
    if frustum_angles.shape != (clusters.shape[0],):
        raise ValueError("frustum_angles must have shape (K,)")

    transforms = torch.eye(4, device=clusters.device, dtype=clusters.dtype).repeat(
        clusters.shape[0], 1, 1
    )
    cosine = torch.cos(frustum_angles).to(dtype=clusters.dtype)
    sine = torch.sin(frustum_angles).to(dtype=clusters.dtype)
    transforms[:, 0, 0] = cosine
    transforms[:, 0, 2] = sine
    transforms[:, 2, 0] = -sine
    transforms[:, 2, 2] = cosine
    aligned = torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        clusters, transforms, False, 1e-6
    )
    valid = torch.arange(clusters.shape[1], device=clusters.device).unsqueeze(0) < counts.unsqueeze(1)
    return aligned * valid.unsqueeze(-1).to(dtype=aligned.dtype)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("faraway_frustum_axis_alignment", _faraway_frustum_axis_alignment_impl)


def _faraway_frustum_axis_alignment_meta(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    frustum_angles: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(clusters, device="meta")


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("faraway_frustum_axis_alignment", _faraway_frustum_axis_alignment_meta)


def frustum_axis_alignment(
    clusters: torch.Tensor,
    counts: torch.Tensor,
    frustum_angles: torch.Tensor,
) -> torch.Tensor:
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_frustum_axis_alignment(
        clusters, counts, frustum_angles
    )
