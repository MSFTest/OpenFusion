# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有几何与回归结果三维框组装算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_assemble_3d_box_impl(
    centroids: torch.Tensor,
    regression_outputs: torch.Tensor,
) -> torch.Tensor:
    """融合几何质心与学习回归结果并生成三维框。

    Args:
        centroids: ``(B, 3)``，相机坐标系质心 ``(x, y, z)``。
        regression_outputs: ``(B, 5)``，依次为
            ``(delta_z, width, length, height, alpha)``。

    Returns:
        boxes_3d: ``(B, 7)``，格式为
            ``(x, y, z_refined, width, length, height, alpha)``。
    """
    if centroids.dim() != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must have shape (B, 3)")
    if regression_outputs.dim() != 2 or regression_outputs.shape != (centroids.shape[0], 5):
        raise ValueError("regression_outputs must have shape (B, 5)")
    refined_depth = centroids[:, 2:3] + regression_outputs[:, 0:1]
    return torch.ops.fusionrepo.feature_concatenation(
        [centroids[:, :2], refined_depth, regression_outputs[:, 1:]], 1
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("faraway_assemble_3d_box", _faraway_assemble_3d_box_impl)


def _faraway_assemble_3d_box_meta(
    centroids: torch.Tensor,
    regression_outputs: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((centroids.shape[0], 7), device="meta", dtype=centroids.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("faraway_assemble_3d_box", _faraway_assemble_3d_box_meta)


def assemble_faraway_3d_box(
    centroids: torch.Tensor,
    regression_outputs: torch.Tensor,
) -> torch.Tensor:
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_assemble_3d_box(
        centroids, regression_outputs
    )
