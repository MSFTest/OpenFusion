# -*- coding: utf-8 -*-
"""
batched_homogeneous_transform_and_normalize

说明：
    批量齐次坐标变换与可选向量归一化算子。

    该算子用于将三维点转换为齐次坐标后应用 4x4 变换矩阵，
    可选择将输出三维向量按 L2 范数归一化为单位方向向量。

Args:
    points:
        三维点坐标，shape 为 (M, 3) 或 (B, M, 3)。
        当输入为 (M, 3) 且 transform 为 (B, 4, 4) 时，会广播到 B 个 batch。

    transform:
        齐次变换矩阵，shape 为 (4, 4) 或 (B, 4, 4)。
        矩阵语义由调用方保证，可表示世界坐标到相机坐标、
        LiDAR 坐标到相机坐标、体素中心到目标坐标等。

    normalize:
        是否对输出三维向量做 L2 归一化。3D-CVF 的 BEV 到相机 ray 构造需要 True。

    eps:
        归一化时的最小范数，避免除零。

Returns:
    transformed:
        变换后的三维点或方向向量，shape 为 (B, M, 3)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _batched_homogeneous_transform_and_normalize_impl(
    points: torch.Tensor,
    transform: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if points.dim() not in (2, 3) or points.shape[-1] != 3:
        raise ValueError("points must have shape (M, 3) or (B, M, 3)")
    if transform.dim() not in (2, 3) or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape (4, 4) or (B, 4, 4)")

    if points.dim() == 2:
        points_batched = points.unsqueeze(0)
    else:
        points_batched = points

    if transform.dim() == 2:
        transform_batched = transform.unsqueeze(0)
    else:
        transform_batched = transform

    batch_points = points_batched.shape[0]
    batch_transform = transform_batched.shape[0]
    batch_size = max(batch_points, batch_transform)

    if batch_points not in (1, batch_size):
        raise ValueError("points batch dimension is not broadcastable")
    if batch_transform not in (1, batch_size):
        raise ValueError("transform batch dimension is not broadcastable")

    points_batched = points_batched.expand(batch_size, -1, -1)
    transform_batched = transform_batched.expand(batch_size, -1, -1)

    ones = torch.ones(
        (*points_batched.shape[:-1], 1),
        device=points.device,
        dtype=points.dtype,
    )
    points_homo = torch.cat([points_batched, ones], dim=-1)
    transformed_homo = torch.matmul(transform_batched, points_homo.transpose(1, 2))
    transformed = transformed_homo[:, :3, :].transpose(1, 2).contiguous()

    if normalize:
        norm = transformed.norm(dim=-1, keepdim=True).clamp_min(float(eps))
        transformed = transformed / norm

    return transformed


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "batched_homogeneous_transform_and_normalize",
    _batched_homogeneous_transform_and_normalize_impl,
)


def _batched_homogeneous_transform_and_normalize_meta(
    points: torch.Tensor,
    transform: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if points.dim() == 2:
        batch_points = 1
        num_points = points.shape[0]
    else:
        batch_points = points.shape[0]
        num_points = points.shape[1]

    batch_transform = 1 if transform.dim() == 2 else transform.shape[0]
    batch_size = max(batch_points, batch_transform)

    return torch.empty(
        (batch_size, num_points, 3),
        device="meta",
        dtype=points.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "batched_homogeneous_transform_and_normalize",
    _batched_homogeneous_transform_and_normalize_meta,
)


def batched_homogeneous_transform_and_normalize(
    points: torch.Tensor,
    transform: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.ops.fusionrepo.batched_homogeneous_transform_and_normalize(
        points,
        transform,
        bool(normalize),
        float(eps),
    )
