# -*- coding: utf-8 -*-
"""
PointProjectionModule

将 LiDAR 点云投影到图像平面。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import utils  # noqa: F401


class PointProjectionModule(nn.Module):
    """
    PointPainting 点投影模块。

    输入:
        points_lidar: (N, 4) LiDAR 点云，格式为 (x, y, z, r)。
        tr_velo2cam: (4, 4) LiDAR 到相机坐标系变换矩阵。
        proj_matrix: (4, 4) 或 (3, 4) 相机投影矩阵。
        r0_rect: (4, 4) 或 (3, 3) 校正矩阵。
        image_shape: (H, W)，用于生成有效投影 mask。

    输出:
        dict:
            points_camera: (N, 4)
            pixel_coords: (N, 2)
            valid_mask: (N,)
    """

    def forward(
        self,
        points_lidar: torch.Tensor,
        tr_velo2cam: torch.Tensor,
        proj_matrix: torch.Tensor,
        r0_rect: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> dict:
        tr_velo2cam = tr_velo2cam.to(device=points_lidar.device, dtype=points_lidar.dtype)
        proj_matrix = proj_matrix.to(device=points_lidar.device, dtype=points_lidar.dtype)
        r0_rect = r0_rect.to(device=points_lidar.device, dtype=points_lidar.dtype)
        points_camera = torch.ops.fusionrepo.pointpainting_lidar_to_camera_transform(
            points_lidar,
            tr_velo2cam,
        )
        pixel_coords = torch.ops.fusionrepo.pointpainting_camera_to_image_projection(
            points_camera,
            proj_matrix,
            r0_rect,
            1e-6,
        )

        valid_mask = points_camera[:, 2] > 0
        if image_shape is not None:
            height, width = int(image_shape[0]), int(image_shape[1])
            u = pixel_coords[:, 0]
            v = pixel_coords[:, 1]
            valid_mask = valid_mask & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        return {
            "points_camera": points_camera,
            "pixel_coords": pixel_coords,
            "valid_mask": valid_mask,
        }
