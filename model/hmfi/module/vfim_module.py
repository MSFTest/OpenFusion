# -*- coding: utf-8 -*-
"""
VFIMModule - Voxel Feature Interaction Module

Object-level consistency loss between homogeneous LiDAR voxel features and
image voxel features. This is a lightweight implementation of the HMFI VFIM
training signal: ROI pool both modalities, project them to a shared embedding,
and optimize a symmetric cosine similarity loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VFIMModule(nn.Module):
    def __init__(
        self,
        lidar_channels: int,
        image_channels: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.lidar_projector = nn.Sequential(
            nn.Linear(lidar_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.image_projector = nn.Sequential(
            nn.Linear(image_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lidar_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.image_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _roi_mean_pool(
        self,
        rois: torch.Tensor,
        voxel_centers: torch.Tensor,
        voxel_features: torch.Tensor
    ) -> torch.Tensor:
        gathered, valid_counts = torch.ops.fusionrepo.gather_roi_features(
            rois,
            voxel_centers,
            voxel_features
        )
        if gathered.numel() == 0:
            return voxel_features.new_zeros((0, voxel_features.shape[-1]))

        pooled = gathered.sum(dim=1) / valid_counts.clamp(min=1).unsqueeze(-1).to(gathered.dtype)
        valid = valid_counts > 0
        return pooled[valid]

    def forward(
        self,
        rois: torch.Tensor,
        voxel_centers: torch.Tensor,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        if rois is None or rois.numel() == 0:
            return lidar_features.sum() * 0.0
        if lidar_features.numel() == 0 or image_features.numel() == 0:
            return lidar_features.sum() * 0.0 + image_features.sum() * 0.0

        n = min(voxel_centers.shape[0], lidar_features.shape[0], image_features.shape[0])
        voxel_centers = voxel_centers[:n]
        lidar_features = lidar_features[:n]
        image_features = image_features[:n]

        lidar_roi = self._roi_mean_pool(rois, voxel_centers, lidar_features)
        image_roi = self._roi_mean_pool(rois, voxel_centers, image_features)
        m = min(lidar_roi.shape[0], image_roi.shape[0])
        if m <= 1:
            return lidar_features.sum() * 0.0 + image_features.sum() * 0.0

        z_lidar = self.lidar_projector(lidar_roi[:m])
        z_image = self.image_projector(image_roi[:m])
        p_lidar = self.lidar_predictor(z_lidar)
        p_image = self.image_predictor(z_image)

        loss_li = 1.0 - F.cosine_similarity(p_lidar, z_image.detach(), dim=1).mean()
        loss_il = 1.0 - F.cosine_similarity(p_image, z_lidar.detach(), dim=1).mean()
        return 0.5 * (loss_li + loss_il)
