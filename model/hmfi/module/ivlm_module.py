# -*- coding: utf-8 -*-
"""
IVLMModule - Image Voxel Lifting Module

使用与 LiDAR voxel center 对齐的内存高效 IVLM 流程。

流程：
    1. generate_depth_bins + expand_depth_bins_to_image → 深度bins
    2. voxel center 通过标定矩阵投影到图像特征平面
    3. 双线性采样图像特征
    4. 根据 LID depth bins 计算深度有效性和最近 bin 权重
    5. 输出与 LiDAR non-empty voxel 一一对齐的 image voxel token
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IVLMModule(nn.Module):
    """
    Image Voxel Lifting Module (IVLM)

    将2D图像特征提升到3D体素空间，与点云特征对齐。

    论文对齐：
        - 深度离散：LID模式 (mode=1)
        - 三线性插值：双线性采样 + 深度方向线性插值
        - 下采样：max pooling，factor=4

    Args:
        depth_mode: 深度离散模式，1=LID（论文默认）
        downsample_factor: 下采样因子，默认4
    """

    def __init__(
        self,
        depth_mode: int = 1,
        downsample_factor: int = 4,
        downsample_mode: int = 0,  # 0=max, 1=avg
        padding_mode: str = "zeros"
    ):
        super().__init__()
        self.depth_mode = depth_mode
        self.downsample_factor = downsample_factor
        self.downsample_mode = downsample_mode
        self.padding_mode = padding_mode

    def forward(
        self,
        image_features: torch.Tensor,
        voxel_centers: torch.Tensor,
        lidar_to_cam: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        depth_range: torch.Tensor,
        num_depth_bins: int,
        image_shape: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            image_features: (B, C, Hf, Wf) 图像 backbone 特征
            voxel_centers: (N, 3) 体素中心坐标 (x, y, z)
            lidar_to_cam: (4, 4) LiDAR→相机 外参
            cam_intrinsic: (3, 3) 相机内参
            depth_range: (2,) [d_min, d_max]
            num_depth_bins: int 深度bin数量
            image_shape: 输入图像张量尺寸 (H_img, W_img)，用于映射到特征平面

        Returns:
            image_voxel_features: (N, C) 与 LiDAR voxel 对齐的图像体素特征
        """
        B, C, Hf, Wf = image_features.shape
        if B != 1:
            raise ValueError("Current HMFI IVLM expects one frame per forward call")

        if voxel_centers.numel() == 0:
            return image_features.new_zeros((0, C))

        # ----------------------------------------
        # Step 1: 生成深度bins
        # ----------------------------------------
        depth_bins_1d = torch.ops.fusionrepo.generate_depth_bins(
            depth_range,
            num_depth_bins,
            self.depth_mode
        )  # (R,)

        # ----------------------------------------
        # Step 2 & 3: 体素中心 → Frustum坐标 (u, v, d)
        # ----------------------------------------
        proj_matrix = torch.ops.fusionrepo.build_projection_matrix(
            cam_intrinsic, lidar_to_cam
        )  # (3, 4)

        pixel_coords = torch.ops.fusionrepo.points_to_image(
            voxel_centers, proj_matrix
        )  # (N, 2)

        frustum_coords = torch.ops.fusionrepo.append_camera_depth(
            voxel_centers, lidar_to_cam, pixel_coords
        )  # (N, 3) = (u, v, d)

        N = voxel_centers.shape[0]

        if image_shape is None:
            image_h = Hf
            image_w = Wf
        else:
            image_h, image_w = int(image_shape[0]), int(image_shape[1])

        # 原图像素坐标映射到特征平面坐标。mmdet FPN 特征通常来自 pad 后图像，
        # 这里用输入张量尺寸估算 stride，可避免直接把原图像素坐标当特征坐标。
        u = frustum_coords[:, 0]
        v = frustum_coords[:, 1]
        d = frustum_coords[:, 2]

        u_feat = u * (float(Wf) / max(float(image_w), 1.0))
        v_feat = v * (float(Hf) / max(float(image_h), 1.0))

        if Wf > 1:
            u_norm = 2.0 * u_feat / (Wf - 1) - 1.0
        else:
            u_norm = torch.zeros_like(u)

        if Hf > 1:
            v_norm = 2.0 * v_feat / (Hf - 1) - 1.0
        else:
            v_norm = torch.zeros_like(v)

        valid_mask = (
            torch.isfinite(u_norm)
            & torch.isfinite(v_norm)
            & torch.isfinite(d)
            & (u_feat >= 0)
            & (u_feat <= max(Wf - 1, 0))
            & (v_feat >= 0)
            & (v_feat <= max(Hf - 1, 0))
            & (d >= depth_range[0])
            & (d <= depth_range[1])
        )

        u_norm = torch.where(torch.isfinite(u_norm), u_norm, torch.zeros_like(u_norm))
        v_norm = torch.where(torch.isfinite(v_norm), v_norm, torch.zeros_like(v_norm))
        u_norm = u_norm.clamp(-1.0, 1.0)
        v_norm = v_norm.clamp(-1.0, 1.0)

        # 构建 2D grid
        grid_2d = torch.stack([u_norm, v_norm], dim=-1)  # (N, 2)
        grid_2d = grid_2d.view(1, N, 1, 2)

        sampled = F.grid_sample(
            image_features,
            grid_2d,
            mode="bilinear",
            align_corners=True,
            padding_mode=self.padding_mode
        )  # (1, C, N, 1)

        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # (N, C)

        # LID 深度权重：越接近离散 depth bin，图像 voxel 特征置信度越高。
        # 这里不改变 token 数量，避免破坏与 LiDAR voxel 的一一对应。
        nearest_dist = torch.abs(d.unsqueeze(1) - depth_bins_1d.view(1, -1)).min(dim=1).values
        if num_depth_bins > 1:
            bin_width = torch.diff(depth_bins_1d).abs().mean().clamp_min(1e-6)
        else:
            bin_width = (depth_range[1] - depth_range[0]).abs().clamp_min(1e-6)
        depth_weight = (1.0 - nearest_dist / bin_width).clamp(0.0, 1.0)
        valid_weight = valid_mask.to(sampled.dtype) * depth_weight.to(sampled.dtype)
        image_voxel_features = sampled * valid_weight.unsqueeze(1)

        return image_voxel_features
