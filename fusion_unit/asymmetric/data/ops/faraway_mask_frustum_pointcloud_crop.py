# -*- coding: utf-8 -*-
"""Faraway-Frustum 私有图像实例到点云 Frustum 裁剪算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _faraway_mask_frustum_pointcloud_crop_impl(
    points_2d_img: torch.Tensor,
    points_3d_cam0: torch.Tensor,
    boxes_2d: torch.Tensor,
    masks_2d: torch.Tensor,
    max_points: int = 2048,
):
    """根据图像实例 mask 或二维框裁剪相机坐标系点云。

    Args:
        points_2d_img: ``(N, 2)``，点在图像中的像素坐标 ``(u, v)``。
        points_3d_cam0: ``(N, 3)``，对应的相机坐标系三维点。
        boxes_2d: ``(K, 4)``，格式为 ``(x1, y1, x2, y2)``。
        masks_2d: ``(K, H, W)``；空张量表示使用二维框。
        max_points: 每个实例保留的最大点数 P。

    Returns:
        clusters: ``(K, P, 3)``，按实例打包的三维点。
        counts: ``(K,)``，每个实例的有效点数。
    """
    if points_2d_img.dim() != 2 or points_2d_img.shape[1] != 2:
        raise ValueError("points_2d_img must have shape (N, 2)")
    if points_3d_cam0.dim() != 2 or points_3d_cam0.shape[1] < 3:
        raise ValueError("points_3d_cam0 must have shape (N, 3+)")
    if points_2d_img.shape[0] != points_3d_cam0.shape[0]:
        raise ValueError("2D and 3D points must have the same N")
    if boxes_2d.dim() != 2 or boxes_2d.shape[1] != 4:
        raise ValueError("boxes_2d must have shape (K, 4)")
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    instance_count = boxes_2d.shape[0]
    clusters = points_3d_cam0.new_zeros((instance_count, int(max_points), 3))
    counts = torch.zeros((instance_count,), device=points_3d_cam0.device, dtype=torch.long)
    use_masks = masks_2d.numel() > 0

    if use_masks and (masks_2d.dim() != 3 or masks_2d.shape[0] != instance_count):
        raise ValueError("masks_2d must have shape (K, H, W) or be empty")

    for index in range(int(instance_count)):
        if use_masks:
            height, width = masks_2d[index].shape
            rounded = torch.round(points_2d_img)
            in_image = (
                (rounded[:, 0] >= 0)
                & (rounded[:, 0] < width)
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < height)
            )
            sampled = torch.ops.fusionrepo.feature_gather_by_pixel_index(
                points_2d_img, masks_2d[index], "round", "HW", True
            )
            selected = in_image & (sampled > 0)
        else:
            box = boxes_2d[index].clone()
            positive_infinity = torch.full_like(box[2:4], float("inf"))
            box[2:4] = torch.nextafter(box[2:4], positive_infinity)
            selected = torch.ops.fusionrepo.points_in_2d_boxes_mask(
                points_2d_img, box, 1, 1, False
            )

        group_ids = torch.where(
            selected,
            torch.zeros_like(selected, dtype=torch.long),
            torch.full_like(selected, -1, dtype=torch.long),
        )
        grouped, grouped_mask = torch.ops.fusionrepo.group_points_by_indices(
            points_3d_cam0[:, :3], group_ids, 1, int(max_points), False
        )
        clusters[index] = grouped[0]
        counts[index] = grouped_mask[0].sum()
    return clusters, counts


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "faraway_mask_frustum_pointcloud_crop",
    _faraway_mask_frustum_pointcloud_crop_impl,
)


def _faraway_mask_frustum_pointcloud_crop_meta(
    points_2d_img: torch.Tensor,
    points_3d_cam0: torch.Tensor,
    boxes_2d: torch.Tensor,
    masks_2d: torch.Tensor,
    max_points: int = 2048,
):
    return (
        torch.empty(
            (boxes_2d.shape[0], int(max_points), 3),
            device="meta",
            dtype=points_3d_cam0.dtype,
        ),
        torch.empty((boxes_2d.shape[0],), device="meta", dtype=torch.long),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "faraway_mask_frustum_pointcloud_crop",
    _faraway_mask_frustum_pointcloud_crop_meta,
)


def mask_frustum_pointcloud_crop(
    points_2d_img: torch.Tensor,
    points_3d_cam0: torch.Tensor,
    boxes_2d: torch.Tensor,
    masks_2d: torch.Tensor,
    max_points: int = 2048,
):
    """Python wrapper，参数及 shape 与实现函数一致。"""
    return torch.ops.fusionrepo.faraway_mask_frustum_pointcloud_crop(
        points_2d_img,
        points_3d_cam0,
        boxes_2d,
        masks_2d,
        int(max_points),
    )
