# -*- coding: utf-8 -*-
"""
HMFIFramework

Hierarchical Multi-modal Fusion with Integration (HMFI) 框架调度层

功能：
    1. 调用图像特征提取接口生成 image features
    2. 调用点云感知接口生成点云特征 / voxel 信息 / 3D proposals
    3. 调用各阶段模块完成：
        - IVLM: 将2D图像特征提升到3D体素空间
        - QFM: 使用LiDAR特征作为Query进行跨模态注意力融合
    4. 提供 train / test 接口

注意：
    本文件不直接调用任何 fusionrepo 算子
    所有融合逻辑封装在 module 中
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

# Backbone 接口
from backbone.image_feature_interface import ImageFeatureInterface
from backbone.point_cloud_feature_interface import PointCloudFeatureInterface

# 模块
from model.hmfi.module.ivlm_module import IVLMModule
from model.hmfi.module.qfm_module import QFMModule
from model.hmfi.module.hmfi_detection_head import HMFIDetectionHead
from model.hmfi.module.vfim_module import VFIMModule


class HMFIFramework(nn.Module):
    """
    HMFI 融合框架

    论文对齐：
        - IVLM: 图像体素提升模块，使用深度离散化 + 三线性插值
        - QFM: 查询融合模块，使用多头注意力进行跨模态融合

    Args:
        image_backbone: 图像特征提取接口
        point_cloud_backbone: 点云感知接口
        lidar_channels: LiDAR特征通道数
        image_channels: 图像体素特征通道数
        fusion_out_channels: 融合输出通道数
        d_k: Query/Key维度
        d_v: Value维度
        num_heads: 注意力头数
        depth_mode: 深度离散模式，1=LID（论文默认）
        downsample_factor: 下采样因子，默认4
        depth_range: 深度范围 [d_min, d_max]
        num_depth_bins: 深度bin数量
    """

    def __init__(
        self,
        image_backbone: ImageFeatureInterface,
        point_cloud_backbone: PointCloudFeatureInterface,
        lidar_channels: int,
        image_channels: int,
        fusion_out_channels: int = 128,
        d_k: int = 64,
        d_v: int = 64,
        num_heads: int = 4,
        depth_mode: int = 1,
        downsample_factor: int = 4,
        depth_range: tuple = (-0.5, 70.0),
        num_depth_bins: int = 128,
        voxel_size: Optional[tuple] = None,
        point_cloud_range: Optional[tuple] = None,
        use_vfim: bool = True,
        num_classes: int = 1,
        box_dim: int = 7
    ):
        super().__init__()

        self.image_backbone = image_backbone
        self.point_cloud_backbone = point_cloud_backbone

        self.lidar_channels = lidar_channels
        self.image_channels = image_channels
        self.fusion_out_channels = fusion_out_channels
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.depth_mode = depth_mode
        self.downsample_factor = downsample_factor
        self.depth_range = depth_range
        self.num_depth_bins = num_depth_bins
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_vfim = use_vfim
        self.num_classes = num_classes
        self.box_dim = box_dim

        # ----------------------------------------
        # IVLM: Image Voxel Lifting Module
        # ----------------------------------------
        self.ivlm_module = IVLMModule(
            depth_mode=depth_mode,
            downsample_factor=downsample_factor
        )

        # ----------------------------------------
        # QFM: Query Fusion Module
        # ----------------------------------------
        self.qfm_module = QFMModule(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            d_k=d_k,
            d_v=d_v,
            num_heads=num_heads,
            fusion_out_channels=fusion_out_channels
        )

        # ----------------------------------------
        # Detection Head
        # ----------------------------------------
        self.detection_head = HMFIDetectionHead(
            in_channels=fusion_out_channels,
            num_classes=num_classes,
            box_dim=box_dim
        )

        self.vfim_module = VFIMModule(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            hidden_dim=256
        ) if use_vfim else None

    def _voxel_coords_to_lidar_centers(
        self,
        voxel_coords: torch.Tensor,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        if voxel_coords is None:
            return None
        if self.voxel_size is None or self.point_cloud_range is None:
            return None
        if voxel_coords.dim() != 2 or voxel_coords.shape[1] not in (3, 4):
            return None

        coords = voxel_coords[:, 1:4] if voxel_coords.shape[1] == 4 else voxel_coords
        coords = coords.to(device=device, dtype=torch.float32)
        xyz_indices = torch.stack([coords[:, 2], coords[:, 1], coords[:, 0]], dim=1)
        voxel_size = torch.tensor(self.voxel_size, dtype=torch.float32, device=device)
        pc_min = torch.tensor(self.point_cloud_range[:3], dtype=torch.float32, device=device)
        return (xyz_indices + 0.5) * voxel_size + pc_min

    def _select_lidar_features(
        self,
        pc_out: dict,
        voxel_centers: torch.Tensor
    ) -> torch.Tensor:
        voxel_features = pc_out.get("voxel_features", None)
        if voxel_features is None:
            voxel_features = pc_out.get("pts_feats", None)
        if voxel_features is None:
            raise ValueError("pts_feats is required from point_cloud_backbone")

        if voxel_features.dim() != 2:
            raise ValueError(
                f"HMFI requires 2D lidar token features, got {tuple(voxel_features.shape)} "
                f"with feature_layout={pc_out.get('feature_layout')}"
            )

        if voxel_features.shape[0] != voxel_centers.shape[0]:
            n = min(voxel_features.shape[0], voxel_centers.shape[0])
            voxel_features = voxel_features[:n]
        return voxel_features

    def _forward_impl(self, batch: dict, rescale: bool):
        images = batch["images"]
        points = batch["points"]
        img_metas = batch.get("img_metas", None)
        proj_matrix = batch["proj_matrix"]
        lidar_to_cam = batch.get("lidar_to_cam", None)
        cam_intrinsic = batch.get("cam_intrinsic", None)

        # 1) 图像特征提取
        image_feat = self.image_backbone(images)  # (B, C, H, W)

        # 2) 点云感知（点云特征提取 + 3D proposals）
        pc_out = self.point_cloud_backbone(points=points)

        voxel_coords = pc_out.get("voxel_coords", None)  # (M, 4) [batch, z, y, x]
        voxel_features = pc_out.get("pts_feats", None)    # (M, C_lidar)
        voxels = pc_out.get("voxels", None)  # (M, T, 3)
        voxel_num_points = pc_out.get("voxel_num_points", None)  # (M,)
        flat_rois = pc_out.get("flat_rois", None)
        flat_roi_batch_indices = pc_out.get("flat_roi_batch_indices", None)
        flat_roi_scores = pc_out.get("flat_roi_scores", None)

        if voxel_features is None:
            raise ValueError("pts_feats is required from point_cloud_backbone")

        # 计算体素中心
        voxel_centers = pc_out.get("voxel_centers_lidar", None)
        if voxel_centers is not None:
            voxel_centers = voxel_centers.to(device=images.device, dtype=torch.float32)
        elif voxel_coords is not None and voxel_coords.shape[1] in (3, 4):
            voxel_centers = self._voxel_coords_to_lidar_centers(voxel_coords, images.device)
            if voxel_centers is None:
                # Last-resort compatibility fallback. This keeps old models runnable,
                # but HMFI accuracy requires voxel_size and point_cloud_range.
                voxel_centers = voxel_coords[:, -3:].to(device=images.device, dtype=torch.float32)
        elif voxels is not None and voxel_num_points is not None:
            # 使用 compute_voxel_centroid 从体素内点计算质心
            voxel_centers = torch.ops.fusionrepo.compute_voxel_centroid(
                voxels, voxel_num_points
            )  # (M, 3)
        else:
            raise ValueError("voxel_coords or (voxels, voxel_num_points) is required")

        # 确保 voxel_centers 是浮点类型
        if not voxel_centers.is_floating_point():
            voxel_centers = voxel_centers.float()
        voxel_centers = voxel_centers.to(device=images.device)

        voxel_features = self._select_lidar_features(pc_out, voxel_centers)
        if voxel_features.shape[0] != voxel_centers.shape[0]:
            n = min(voxel_features.shape[0], voxel_centers.shape[0])
            voxel_features = voxel_features[:n]
            voxel_centers = voxel_centers[:n]
            if voxel_coords is not None:
                voxel_coords = voxel_coords[:n]

        # ----------------------------------------
        # IVLM: 将2D图像特征提升到3D体素空间
        # ----------------------------------------
        # 使用体素中心坐标进行视锥投影
        if lidar_to_cam is None:
            # 默认使用单位矩阵
            lidar_to_cam = torch.eye(4, dtype=torch.float32, device=images.device)
        if cam_intrinsic is None:
            # 从proj_matrix提取内参
            cam_intrinsic = proj_matrix[:3, :3]

        # 确保 cam_intrinsic 是浮点类型
        if not cam_intrinsic.is_floating_point():
            cam_intrinsic = cam_intrinsic.float()

        image_voxel_features = self.ivlm_module(
            image_features=image_feat,
            voxel_centers=voxel_centers,
            lidar_to_cam=lidar_to_cam,
            cam_intrinsic=cam_intrinsic,
            depth_range=torch.tensor(self.depth_range, dtype=torch.float32, device=images.device),
            num_depth_bins=self.num_depth_bins,
            image_shape=tuple(images.shape[-2:])
        )  # (N', C_img)
        if self.downsample_factor > 1 and image_voxel_features.shape[0] > self.downsample_factor:
            qfm_image_features = image_voxel_features[::self.downsample_factor].contiguous()
        else:
            qfm_image_features = image_voxel_features

        # ----------------------------------------
        # QFM: 使用LiDAR特征作为Query进行跨模态注意力融合
        # ----------------------------------------
        fused_features = self.qfm_module(
            lidar_features=voxel_features,  # (N, C_lidar)
            image_features=qfm_image_features  # (M, C_img)
        )  # (N, fusion_out_channels)

        # ----------------------------------------
        # Detection Head: 使用ROIs和融合特征进行检测
        # ----------------------------------------
        if flat_rois is not None and len(flat_rois) > 0:
            det_out = self.detection_head(
                rois=flat_rois,
                voxel_coords=voxel_centers,
                fused_features=fused_features
            )
        else:
            det_out = {
                "cls_score": torch.zeros((0, self.num_classes), device=fused_features.device),
                "reg_offset": torch.zeros((0, self.box_dim), device=fused_features.device)
            }

        if self.training and self.vfim_module is not None and flat_rois is not None and len(flat_rois) > 0:
            vfim_loss = self.vfim_module(
                rois=flat_rois,
                voxel_centers=voxel_centers,
                lidar_features=voxel_features,
                image_features=image_voxel_features
            )
        else:
            vfim_loss = fused_features.sum() * 0.0

        return {
            "fused_features": fused_features,
            "voxel_features": voxel_features,
            "image_voxel_features": image_voxel_features,
            "voxel_centers": voxel_centers,
            "voxel_coords": voxel_coords,
            "flat_rois": flat_rois,
            "flat_roi_batch_indices": flat_roi_batch_indices,
            "flat_roi_scores": flat_roi_scores,
            "image_features": image_feat,
            "pc_out": pc_out,
            "cls_score": det_out["cls_score"],
            "reg_offset": det_out["reg_offset"],
            "roi_valid_counts": det_out.get("valid_counts", None),
            "vfim_loss": vfim_loss
        }

    def train_forward(self, batch: dict):
        return self._forward_impl(batch=batch, rescale=False)

    @torch.no_grad()
    def test_forward(self, batch: dict):
        self.eval()
        return self._forward_impl(batch=batch, rescale=True)
