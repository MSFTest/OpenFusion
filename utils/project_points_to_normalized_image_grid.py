# -*- coding: utf-8 -*-
"""
多视角三维点到图像归一化采样网格的投影算子。

Args:
    points: 三维点坐标，shape 为 (B, P, L, 3) 或 (B, N, 3)，坐标系由
        transform 语义决定，IS-Fusion 中为增强后的 LiDAR/ego 坐标系。
    transform: 点到图像齐次矩阵，shape 为 (B, V, 4, 4) 或 (B, 4, 4)。
    img_aug_matrix: 图像后处理齐次矩阵，shape 为 (B, V, 4, 4) 或 (B, 4, 4)。
    lidar_aug_matrix: LiDAR 增强齐次矩阵，shape 为 (B, 4, 4)。
    image_size: 图像尺寸，shape 为 (2,) 或 (B, 2)，顺序为 (height, width)。
    return_depth: 是否返回投影深度。

Returns:
    list[Tensor]:
        - normalized_uv: 归一化图像坐标，shape 为 (B, V, ..., 2)，范围 [-1, 1]。
        - valid_mask: 有效投影掩码，shape 为 (B, V, ...)。
        - depth: 当 return_depth=True 时返回，shape 为 (B, V, ...)。
"""

from __future__ import annotations

from typing import List

import torch
from torch.library import Library


def _project_points_to_normalized_image_grid_impl(
    points: torch.Tensor,
    transform: torch.Tensor,
    img_aug_matrix: torch.Tensor,
    lidar_aug_matrix: torch.Tensor,
    image_size: torch.Tensor,
    return_depth: bool = False,
) -> List[torch.Tensor]:
    if points.dim() not in (3, 4) or points.shape[-1] != 3:
        raise ValueError("points must have shape (B,N,3) or (B,P,L,3)")
    if transform.dim() == 3:
        transform = transform.unsqueeze(1)
    if img_aug_matrix.dim() == 3:
        img_aug_matrix = img_aug_matrix.unsqueeze(1)
    if transform.dim() != 4 or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape (B,V,4,4) or (B,4,4)")
    if img_aug_matrix.dim() != 4 or img_aug_matrix.shape[-2:] != (4, 4):
        raise ValueError("img_aug_matrix must have shape (B,V,4,4) or (B,4,4)")
    if lidar_aug_matrix.dim() == 4 and lidar_aug_matrix.shape[1] == 1:
        lidar_aug_matrix = lidar_aug_matrix[:, 0]
    if lidar_aug_matrix.dim() != 3 or lidar_aug_matrix.shape[-2:] != (4, 4):
        raise ValueError("lidar_aug_matrix must have shape (B,4,4) or (B,1,4,4)")

    batch_size = points.shape[0]
    original_shape = points.shape[1:-1]
    flat_points = points.reshape(batch_size, -1, 3)
    num_points = flat_points.shape[1]
    num_views = transform.shape[1]

    ones = torch.ones((*flat_points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([flat_points, ones], dim=-1)
    inv_lidar_aug = torch.linalg.inv(lidar_aug_matrix.to(points.dtype))
    points_h = torch.matmul(inv_lidar_aug[:, None, :, :], points_h[:, :, :, None]).squeeze(-1)

    cam_points = torch.matmul(transform.to(points.dtype)[:, :, None, :, :], points_h[:, None, :, :, None]).squeeze(-1)
    depth = cam_points[..., 2]
    depth_safe = depth.clamp_min(1e-5)
    uv = cam_points[..., :2] / depth_safe.unsqueeze(-1)

    uv_h = torch.cat([uv, torch.ones_like(depth).unsqueeze(-1), torch.ones_like(depth).unsqueeze(-1)], dim=-1)
    uv_aug = torch.matmul(img_aug_matrix.to(points.dtype)[:, :, None, :, :], uv_h[..., None]).squeeze(-1)[..., :2]

    if image_size.dim() == 1:
        height = image_size[0].to(dtype=points.dtype, device=points.device)
        width = image_size[1].to(dtype=points.dtype, device=points.device)
        norm_x = uv_aug[..., 0] / width
        norm_y = uv_aug[..., 1] / height
    elif image_size.dim() == 2:
        size = image_size.to(dtype=points.dtype, device=points.device)[:, None, None, :]
        norm_x = uv_aug[..., 0] / size[..., 1]
        norm_y = uv_aug[..., 1] / size[..., 0]
    else:
        raise ValueError("image_size must have shape (2,) or (B,2)")

    normalized_uv = torch.stack([norm_x * 2.0 - 1.0, norm_y * 2.0 - 1.0], dim=-1)
    valid_mask = (
        (depth > 1e-5)
        & (normalized_uv[..., 0] > -1.0)
        & (normalized_uv[..., 0] < 1.0)
        & (normalized_uv[..., 1] > -1.0)
        & (normalized_uv[..., 1] < 1.0)
    )

    uv_shape = (batch_size, num_views, *original_shape, 2)
    mask_shape = (batch_size, num_views, *original_shape)
    outputs = [
        normalized_uv.reshape(uv_shape).contiguous(),
        valid_mask.reshape(mask_shape).contiguous(),
    ]
    if return_depth:
        outputs.append(depth.reshape(mask_shape).contiguous())
    return outputs


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("project_points_to_normalized_image_grid", _project_points_to_normalized_image_grid_impl)


def _project_points_to_normalized_image_grid_meta(
    points: torch.Tensor,
    transform: torch.Tensor,
    img_aug_matrix: torch.Tensor,
    lidar_aug_matrix: torch.Tensor,
    image_size: torch.Tensor,
    return_depth: bool = False,
) -> List[torch.Tensor]:
    batch_size = points.shape[0]
    num_views = transform.shape[1] if transform.dim() == 4 else 1
    original_shape = tuple(points.shape[1:-1])
    uv_shape = (batch_size, num_views, *original_shape, 2)
    mask_shape = (batch_size, num_views, *original_shape)
    outputs = [
        torch.empty(uv_shape, device="meta", dtype=points.dtype),
        torch.empty(mask_shape, device="meta", dtype=torch.bool),
    ]
    if return_depth:
        outputs.append(torch.empty(mask_shape, device="meta", dtype=points.dtype))
    return outputs


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("project_points_to_normalized_image_grid", _project_points_to_normalized_image_grid_meta)


def project_points_to_normalized_image_grid(
    points: torch.Tensor,
    transform: torch.Tensor,
    img_aug_matrix: torch.Tensor,
    lidar_aug_matrix: torch.Tensor,
    image_size: torch.Tensor,
    return_depth: bool = False,
) -> List[torch.Tensor]:
    return torch.ops.fusionrepo.project_points_to_normalized_image_grid(
        points,
        transform,
        img_aug_matrix,
        lidar_aug_matrix,
        image_size,
        bool(return_depth),
    )
