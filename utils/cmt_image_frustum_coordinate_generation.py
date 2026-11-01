# -*- coding: utf-8 -*-
"""CMT 图像 token 射锥坐标生成算子。"""

from __future__ import annotations

import torch
from torch.library import Library

from utils.batched_homogeneous_transform_and_normalize import (
    batched_homogeneous_transform_and_normalize,
)


def _impl(
    image_feature: torch.Tensor,
    img2lidar: torch.Tensor,
    pc_range: torch.Tensor,
    depth_num: int,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    """按 CMT 源码的像素、深度与矩阵约定生成三维射锥。"""
    if image_feature.ndim != 4:
        raise ValueError("image_feature must have shape [B*V,C,H,W]")
    if img2lidar.ndim != 4 or img2lidar.shape[-2:] != (4, 4):
        raise ValueError("img2lidar must have shape [B,V,4,4]")
    batch, views = img2lidar.shape[:2]
    batch_views, _, height, width = image_feature.shape
    if batch_views != batch * views:
        raise ValueError("image feature leading dimension must equal B*V")
    if height <= 0 or width <= 0 or depth_num <= 0:
        raise ValueError("H, W and depth_num must be positive")
    if pc_range.numel() != 6:
        raise ValueError("pc_range must have 6 values")
    device = image_feature.device
    pc = pc_range.detach().to(device=device, dtype=torch.float32)
    coords_h = torch.arange(height, device=device, dtype=torch.float32) * float(pad_h) / height
    coords_w = torch.arange(width, device=device, dtype=torch.float32) * float(pad_w) / width
    coords_d = 1.0 + torch.arange(depth_num, device=device, dtype=torch.float32) * (
        pc[3] - 1.0
    ) / depth_num
    grid_h, grid_w, grid_d = torch.meshgrid(coords_h, coords_w, coords_d, indexing="ij")
    camera_points = torch.stack((grid_w * grid_d, grid_h * grid_d, grid_d), dim=-1)
    lidar_points = batched_homogeneous_transform_and_normalize(
        camera_points.reshape(-1, 3),
        img2lidar.detach().to(device=device, dtype=torch.float32).reshape(batch_views, 4, 4),
        False,
        1e-6,
    ).reshape(batch_views, height, width, depth_num, 3)
    return (lidar_points - pc[:3]) / (pc[3:] - pc[:3])


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("cmt_image_frustum_coordinates", _impl)


def _meta(image_feature, img2lidar, pc_range, depth_num: int, pad_h: int, pad_w: int):
    del pad_h, pad_w
    if image_feature.ndim != 4:
        raise ValueError("image_feature must have shape [B*V,C,H,W]")
    if img2lidar.ndim != 4 or img2lidar.shape[-2:] != (4, 4):
        raise ValueError("img2lidar must have shape [B,V,4,4]")
    batch_views, _, height, width = image_feature.shape
    if batch_views != img2lidar.shape[0] * img2lidar.shape[1]:
        raise ValueError("image feature leading dimension must equal B*V")
    if height <= 0 or width <= 0 or depth_num <= 0:
        raise ValueError("H, W and depth_num must be positive")
    if pc_range.numel() != 6:
        raise ValueError("pc_range must have 6 values")
    return torch.empty(
        (batch_views, height, width, depth_num, 3), device="meta", dtype=torch.float32
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("cmt_image_frustum_coordinates", _meta)


def cmt_image_frustum_coordinates(
    image_feature: torch.Tensor,
    img2lidar: torch.Tensor,
    pc_range: torch.Tensor,
    depth_num: int,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    """生成图像 token 沿射线的归一化 LiDAR 坐标。

    Args:
        image_feature: ``[(B*V),C,H,W]``，batch-major/camera-minor；仅 shape/device
            参与计算。
        img2lidar: ``[B,V,4,4]``，列向量约定下增强图像坐标到当前 LiDAR
            米制坐标的齐次变换，按原 metadata 语义 detach。
        pc_range: ``[xmin,ymin,zmin,xmax,ymax,zmax]``，单位米。
        depth_num: 每条射线的深度样本数 ``D``。
        pad_h: 输入图像 padding 高度；像素为 ``arange(H)*pad_h/H``。
        pad_w: 输入图像 padding 宽度；像素为 ``arange(W)*pad_w/W``。

    Returns:
        ``float32[(B*V),H,W,D,3]``，与 image_feature 同 device、contiguous；
        深度为 ``1+k*(xmax-1)/D``，结果归一化但不 clamp、不筛除不可见点。
        标定与范围不可微，输出对 image_feature 数值无梯度。
    """
    return torch.ops.fusionrepo.cmt_image_frustum_coordinates(
        image_feature, img2lidar, pc_range, int(depth_num), int(pad_h), int(pad_w)
    )
