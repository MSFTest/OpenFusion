# -*- coding: utf-8 -*-
"""Geometry projection module for EPNet."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import utils  # noqa: F401 - registers fusionrepo operators


class GeometryProjectionModule(nn.Module):
    """
    将 LiDAR 点投影到图像特征平面并采样图像点特征。

    Args:
        align_corners: grid_sample 坐标对齐方式。

    Inputs:
        points: (B, N, C) 或 (N, C)，LiDAR 坐标系点，前三维为 x/y/z。
        image_features: (B, C_img, Hf, Wf)，图像特征图。
        lidar_to_cam_r: (B, 3, 3)，LiDAR 到相机坐标旋转矩阵。
        lidar_to_cam_t: (B, 3) 或 (B, 3, 1)，LiDAR 到相机坐标平移。
        cam_intrinsic: (B, 3, 3)，相机内参。
        image_hw: 原始图像 (height, width)。grid_sample 的归一化坐标按原图坐标系计算。

    Returns:
        dict:
            xyz: (B, N, 3)，LiDAR 坐标点。
            uv: (B, N, 2)，图像像素坐标。
            depth: (B, N)，相机坐标 z 深度。
            grid: (B, N, 1, 2)，grid_sample 归一化坐标。
            valid_mask: (B, N)，在图像范围内且深度为正的点。
            image_point_features: (B, C_img, N)，按点采样的图像特征。
    """

    def __init__(self, align_corners: bool = True):
        super().__init__()
        self.align_corners = bool(align_corners)

    def forward(
        self,
        points: torch.Tensor,
        image_features: torch.Tensor,
        lidar_to_cam_r: torch.Tensor,
        lidar_to_cam_t: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if lidar_to_cam_t.dim() == 2:
            lidar_to_cam_t = lidar_to_cam_t.unsqueeze(-1)

        if image_hw is None:
            height = int(image_features.shape[-2])
            width = int(image_features.shape[-1])
        else:
            height = int(image_hw[0])
            width = int(image_hw[1])
        xyz = torch.ops.fusionrepo.epnet_lidar_point_extractor(points)
        uv, depth, _ = torch.ops.fusionrepo.epnet_lidar_to_image_projector(
            xyz, lidar_to_cam_r, lidar_to_cam_t, cam_intrinsic
        )
        _, grid, valid_mask = torch.ops.fusionrepo.epnet_grid_generator(
            width, height, uv, depth
        )
        sampled = torch.ops.fusionrepo.epnet_image_sampler(image_features, grid)
        return {
            "xyz": xyz,
            "uv": uv,
            "depth": depth,
            "grid": grid,
            "valid_mask": valid_mask,
            "image_point_features": sampled,
        }
