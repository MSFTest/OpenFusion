"""LiDAR voxel encoding branch for BEVFusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from common.bevfusion_lidar_flatten_z_to_bev import bevfusion_lidar_flatten_z_to_bev
from common.bevfusion_sparse3d_backbone import BEVFusionSparse3DBackbone
from common.bevfusion_voxel_feature_encoder import BEVFusionVoxelFeatureEncoder
from utils.bevfusion_lidar_voxelization import bevfusion_lidar_voxelization


class LiDARBEVModule(nn.Module):
    """Voxelize LiDAR points and produce a dense LiDAR-coordinate BEV feature.

    Input:
        batch["points"]: list of ``B`` tensors shaped ``(N_i, 4)`` where the
            first three channels are LiDAR ``(x, y, z)`` and channel four is
            reflectance.

    Output:
        LiDAR BEV tensor ``(B, C_out, Y, X)``.

    Operators:
        ``bevfusion_lidar_voxelization``, ``BEVFusionVoxelFeatureEncoder``,
        ``BEVFusionSparse3DBackbone`` and ``bevfusion_lidar_flatten_z_to_bev``.
    """

    def __init__(
        self,
        point_cloud_range,
        voxel_size,
        grid_size,
        out_channels: int,
        max_points_per_voxel: int = 16,
        max_voxels: int = 12000,
    ):
        super().__init__()
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32))
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)
        self.voxel_encoder = BEVFusionVoxelFeatureEncoder(4, out_channels, out_channels)
        self.sparse_backbone = BEVFusionSparse3DBackbone(
            out_channels, out_channels, out_channels, tuple(grid_size)
        )

    @staticmethod
    def _pad_points(points) -> torch.Tensor:
        max_points = max(int(item.shape[0]) for item in points)
        padded = points[0].new_zeros((len(points), max_points, points[0].shape[1]))
        for index, item in enumerate(points):
            padded[index, : item.shape[0]] = item
        return padded

    def forward(self, batch: dict) -> torch.Tensor:
        points = batch["points"]
        if torch.is_tensor(points):
            padded = points
        else:
            padded = self._pad_points(points)
        voxels, coordinates, counts = bevfusion_lidar_voxelization(
            padded,
            self.voxel_size.to(padded),
            self.point_cloud_range.to(padded),
            self.max_points_per_voxel,
            self.max_voxels,
        )
        voxel_features = self.voxel_encoder(voxels, counts)
        features_3d = self.sparse_backbone(voxel_features, coordinates, padded.shape[0])
        return bevfusion_lidar_flatten_z_to_bev(features_3d)
