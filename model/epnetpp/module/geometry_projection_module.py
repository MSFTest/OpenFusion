# -*- coding: utf-8 -*-
"""EPNet++ LiDAR-to-image projection and point-wise image sampling module."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import utils  # noqa: F401 - registers fusionrepo operators


class GeometryProjectionModule(nn.Module):
    """
    将 LiDAR 点投影到相机图像并采样图像特征。

    坐标系:
        输入 points 前三维为 LiDAR 坐标；`lidar_to_cam_r/t` 和 `cam_intrinsic`
        将 LiDAR 点投影到 camera/image 坐标。输出 `uv` 为原图像素坐标，
        `xy_norm` 为 grid_sample 使用的 [-1, 1] 归一化坐标。

    Args:
        points: Tensor，shape 为 (B, N, C) 或 (N, C)。
        image_features: Tensor，shape 为 (B, C_img, Hf, Wf)。
        lidar_to_cam_r: Tensor，shape 为 (B, 3, 3)。
        lidar_to_cam_t: Tensor，shape 为 (B, 3) 或 (B, 3, 1)。
        cam_intrinsic: Tensor，shape 为 (B, 3, 3)。
        image_hw: 原图尺寸 (H, W)，用于有效点判断。

    Returns:
        dict，包含 xyz、uv、depth、xy_norm、valid_mask、image_point_features。
    """

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

        image_h = int(image_hw[0]) if image_hw is not None else int(image_features.shape[-2])
        image_w = int(image_hw[1]) if image_hw is not None else int(image_features.shape[-1])
        feat_h = int(image_features.shape[-2])
        feat_w = int(image_features.shape[-1])

        xyz = torch.ops.fusionrepo.epnet_lidar_point_extractor(points)
        uv, depth, _ = torch.ops.fusionrepo.epnet_lidar_to_image_projector(
            xyz,
            lidar_to_cam_r,
            lidar_to_cam_t,
            cam_intrinsic,
        )
        xy_norm, _, valid_mask = torch.ops.fusionrepo.epnet_grid_generator(
            image_w,
            image_h,
            uv,
            depth,
        )
        if (feat_h, feat_w) != (image_h, image_w):
            feat_uv = uv.clone()
            feat_uv[..., 0] = feat_uv[..., 0] * float(feat_w) / max(float(image_w), 1.0)
            feat_uv[..., 1] = feat_uv[..., 1] * float(feat_h) / max(float(image_h), 1.0)
            feat_depth = depth
            feat_xy_norm, _, _ = torch.ops.fusionrepo.epnet_grid_generator(
                feat_w,
                feat_h,
                feat_uv,
                feat_depth,
            )
        else:
            feat_xy_norm = xy_norm
        image_point_features = torch.ops.fusionrepo.epnetpp_image_point_sampling(
            image_features,
            feat_xy_norm,
        )
        return {
            "xyz": xyz,
            "uv": uv,
            "depth": depth,
            "xy_norm": xy_norm,
            "feature_xy_norm": feat_xy_norm,
            "valid_mask": valid_mask,
            "image_point_features": image_point_features,
        }
