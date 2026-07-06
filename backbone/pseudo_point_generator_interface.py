# -*- coding: utf-8 -*-
"""
PseudoPointGeneratorInterface

伪点云生成接口 - 使用MiDaS预训练深度估计模型

基于MiDaS (Intel)的预训练深度估计模型，从单目图像估计深度图，
然后结合相机内外参投影到3D空间生成伪点云。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class PseudoPointGeneratorInterface(nn.Module):
    """
    伪点云生成接口 - 使用MiDaS预训练深度估计

    使用MiDaS预训练模型从图像估计深度图，
    然后通过相机内外参投影到3D空间生成伪点云。

    输入:
        images: (B, 3, H, W) 原始图像
        proj_matrix: (B, 4, 4) 投影矩阵
        lidar_to_cam: (B, 4, 4) LiDAR到相机坐标变换
        cam_intrinsic: (B, 3, 3) 相机内参

    输出:
        dict with keys:
            - pseudo_points: (N, 3+C) 伪点云坐标+特征
            - pseudo_coords: (N, 3) 伪点云坐标
            - pseudo_features: (N, C) 伪点云特征
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        feat_dim: int = 256,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        device: str = "cuda:0"
    ):
        """
        Args:
            model_type: MiDaS模型类型 - "DPT_Large", "DPT_Hybrid", "Midas_Hybrid"
            feat_dim: 输出特征维度
            depth_min: 最小深度阈值
            depth_max: 最大深度阈值
            device: 模型设备
        """
        super().__init__()
        self.model_type = model_type
        self.feat_dim = feat_dim
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.device = device

        # 加载预训练MiDaS模型
        self._load_midas_model(model_type, device)

        # 图像特征映射层：将原始图像(3通道)映射到指定维度
        self.img_feat_map = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_dim, 1)
        )

        # 图像特征映射层：将MiDaS输出特征映射到指定维度
        # MiDaS的DPT_Large输出特征维度是1024
        midas_feat_dim = 1024 if "DPT_Large" in model_type else 1024
        self.feat_map = nn.Sequential(
            nn.Conv2d(midas_feat_dim, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feat_dim, 1)
        )

    def _load_midas_model(self, model_type: str, device: str):
        """加载MiDaS预训练模型"""
        import os
        try:
            # 权重文件路径
            cache_dir = os.path.expanduser("~/.cache/torch/hub")
            checkpoint_path = os.path.join(cache_dir, "checkpoints", "dpt_large_384.pt")

            if os.path.exists(checkpoint_path):
                # 先通过hub获取模型结构（不下载），再加载本地权重
                self.midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=False, verbose=False)
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.midas.load_state_dict(state_dict)
                self.midas.to(device)
                self.midas.eval()
                print(f"[PseudoPointGenerator] Loaded MiDaS model from local cache: {checkpoint_path}")

                # 获取MiDaS的transform用于预处理
                try:
                    self.midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms_dpt', verbose=False)
                except:
                    self.midas_transform = None
            else:
                # 尝试从网络加载
                self.midas = torch.hub.load('intel-isl/MiDaS', model_type, pretrained=True, verbose=False)
                self.midas.to(device)
                self.midas.eval()
                print(f"[PseudoPointGenerator] Loaded MiDaS model from hub: {model_type}")

                try:
                    self.midas_transform = torch.hub.load('intel-isl/MiDaS', 'transforms_dpt', verbose=False)
                except:
                    self.midas_transform = None

        except Exception as e:
            print(f"[WARNING] Failed to load MiDaS model: {e}")
            print("[PseudoPointGenerator] Will use fallback depth estimation")
            self.midas = None
            self.midas_transform = None

    def _estimate_depth_midas(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用MiDaS估计深度图

        Args:
            images: (B, 3, H, W) 归一化到[0,1]的图像

        Returns:
            depth: (B, 1, H, W) 深度图
        """
        B, C, H, W = images.shape

        if self.midas is not None:
            with torch.no_grad():
                # 使用MiDaS的transform进行预处理
                if self.midas_transform is not None:
                    # MiDaS的transform期望单张图像 (H, W, 3)
                    # 需要将batch拆开处理
                    depth_list = []
                    for b in range(B):
                        img_b = images[b]  # (3, H, W)
                        # 转为 (H, W, 3)
                        img_bhw3 = img_b.permute(1, 2, 0).cpu().numpy()
                        # 应用transform
                        input_batch = self.midas_transform(img_bhw3).to(images.device)
                        # 估计深度
                        depth_b = self.midas(input_batch)
                        depth_list.append(depth_b)

                    # 堆叠并调整回原始尺寸
                    depth = torch.stack(depth_list, dim=0)  # (B, 1, H', W')
                else:
                    # Fallback: 简单的resize
                    depth = F.interpolate(
                        images.mean(dim=1, keepdim=True),
                        size=(384, 384),
                        mode='bilinear',
                        align_corners=False
                    )
                    depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

                # 确保输出是 (B, 1, H, W)
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)

                return depth
        else:
            # Fallback: 使用简单的基于平面的假设
            depth = torch.ones((B, 1, H, W), device=images.device) * 30.0  # 假设所有点在30米处
            return depth

    def _project_to_3d(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        proj_matrix: Optional[torch.Tensor] = None,
        lidar_to_cam: Optional[torch.Tensor] = None,
        cam_intrinsic: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将深度图投影到3D点云

        Args:
            images: (B, 3, H, W) 图像
            depth: (B, 1, H, W) 深度图
            proj_matrix: (B, 4, 4) 投影矩阵
            lidar_to_cam: (B, 4, 4) LiDAR到相机变换
            cam_intrinsic: (B, 3, 3) 相机内参

        Returns:
            coords: (B, H, W, 3) 3D坐标
        """
        B, C, H, W = images.shape

        # 生成像素网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=images.device, dtype=torch.float32),
            torch.arange(W, device=images.device, dtype=torch.float32),
            indexing='ij'
        )  # (H, W), (H, W)

        # 相机内参（如果没提供，使用标准KITTI内参）
        if cam_intrinsic is None:
            # KITTI标准内参
            fx, fy, cx, cy = 721.5377, 721.5377, 609.5593, 172.854
        else:
            # 处理 cam_intrinsic 形状问题
            if cam_intrinsic.dim() == 2:
                # 如果是 (3, 3)，扩展为 (1, 3, 3)
                cam_intrinsic = cam_intrinsic.unsqueeze(0)
            if cam_intrinsic.dim() == 3 and cam_intrinsic.shape[0] == 1 and B > 1:
                # 如果是 (1, 3, 3) 但 B > 1，扩展到 (B, 3, 3)
                cam_intrinsic = cam_intrinsic.expand(B, -1, -1)
            fx = cam_intrinsic[:, 0, 0]
            fy = cam_intrinsic[:, 1, 1]
            cx = cam_intrinsic[:, 0, 2]
            cy = cam_intrinsic[:, 1, 2]

        # 展开深度
        depth = depth.squeeze(1)  # (B, H, W)

        # 计算3D坐标（相机坐标系下）
        # x = (u - cx) * depth / fx
        # y = (v - cy) * depth / fy
        # z = depth
        x = (x_grid.unsqueeze(0) - cx.view(B, 1, 1)) * depth / fx.view(B, 1, 1)
        y = (y_grid.unsqueeze(0) - cy.view(B, 1, 1)) * depth / fy.view(B, 1, 1)
        z = depth

        coords_cam = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)

        # 如果有lidar_to_cam变换，转换到LiDAR坐标系
        if lidar_to_cam is not None:
            # 处理 lidar_to_cam 形状问题
            if lidar_to_cam.dim() == 2:
                # 如果是 (4, 4)，扩展为 (1, 4, 4)
                lidar_to_cam = lidar_to_cam.unsqueeze(0)
            if lidar_to_cam.dim() == 3 and lidar_to_cam.shape[0] == 1 and B > 1:
                # 如果是 (1, 4, 4) 但 B > 1，扩展到 (B, 4, 4)
                lidar_to_cam = lidar_to_cam.expand(B, -1, -1)

            # coords_lidar = T @ coords_cam (T = cam_to_lidar)
            # 先求逆得到 cam_to_lidar
            cam_to_lidar = torch.inverse(lidar_to_cam)  # (B, 4, 4)

            # 扩展到齐次坐标
            ones = torch.ones((*coords_cam.shape[:3], 1), device=images.device, dtype=coords_cam.dtype)
            coords_homo = torch.cat([coords_cam, ones], dim=-1)  # (B, H, W, 4)

            # 变换
            coords_lidar = torch.einsum('bij,bhwj->bhwi', cam_to_lidar[:, :3, :], coords_homo)

            return coords_lidar
        else:
            return coords_cam

    def forward(
        self,
        images: torch.Tensor,
        proj_matrix: Optional[torch.Tensor] = None,
        lidar_to_cam: Optional[torch.Tensor] = None,
        cam_intrinsic: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) 原始图像 [0, 1] 归一化
            proj_matrix: (B, 4, 4) 投影矩阵
            lidar_to_cam: (B, 4, 4) LiDAR到相机变换
            cam_intrinsic: (B, 3, 3) 相机内参

        Returns:
            dict with pseudo_points, pseudo_coords, pseudo_features
        """
        B, C, H, W = images.shape

        # 1. 使用MiDaS估计深度图
        depth = self._estimate_depth_midas(images)  # (B, 1, H', W') MiDaS输出分辨率

        # 确保depth和images尺寸一致（MiDaS输出可能与原图不同）
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(
                depth,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # (B, 1, H, W)

        # 2. 投影到3D坐标
        coords_3d = self._project_to_3d(
            images, depth,
            proj_matrix=proj_matrix,
            lidar_to_cam=lidar_to_cam,
            cam_intrinsic=cam_intrinsic
        )  # (B, H, W, 3)

        # 3. 提取图像特征用于伪点云
        # 使用图像特征映射层将3通道图像映射到feat_dim通道
        img_features = self.img_feat_map(images)  # (B, feat_dim, H, W)

        # 将 (B, C, H, W) 转换为 (B, H*W, C)
        img_features = img_features.permute(0, 2, 3, 1).reshape(B, H * W, self.feat_dim)

        # 4. 过滤有效深度范围内的点
        depth_flat = depth.squeeze(1).view(B, -1)  # (B, H*W)
        valid_mask = (depth_flat >= self.depth_min) & (depth_flat <= self.depth_max)  # (B, H*W)

        # 5. 展平
        coords_flat = coords_3d.view(B, -1, 3)  # (B, H*W, 3)
        feat_flat = img_features  # (B, H*W, feat_dim)

        # 6. 组合伪点云
        pseudo_points_list = []
        pseudo_coords_list = []
        pseudo_features_list = []

        for b in range(B):
            mask = valid_mask[b]
            n_valid = mask.sum().item()

            if n_valid == 0:
                continue

            coords_b = coords_flat[b, mask]  # (N, 3)
            feat_b = feat_flat[b, mask]  # (N, feat_dim)

            # 拼接坐标和特征
            pseudo_point_b = torch.cat([coords_b, feat_b], dim=-1)  # (N, 3+feat_dim)

            pseudo_points_list.append(pseudo_point_b)
            pseudo_coords_list.append(coords_b)
            pseudo_features_list.append(feat_b)

        if len(pseudo_points_list) == 0:
            device = images.device
            return {
                "pseudo_points": torch.zeros((0, 3 + self.feat_dim), device=device, dtype=torch.float32),
                "pseudo_coords": torch.zeros((0, 3), device=device, dtype=torch.float32),
                "pseudo_features": torch.zeros((0, self.feat_dim), device=device, dtype=torch.float32)
            }

        # 拼接所有batch的点
        pseudo_points = torch.cat(pseudo_points_list, dim=0)  # (N, 3+feat_dim)
        pseudo_coords = torch.cat(pseudo_coords_list, dim=0)  # (N, 3)
        pseudo_features = torch.cat(pseudo_features_list, dim=0)  # (N, feat_dim)

        return {
            "pseudo_points": pseudo_points,
            "pseudo_coords": pseudo_coords,
            "pseudo_features": pseudo_features
        }
