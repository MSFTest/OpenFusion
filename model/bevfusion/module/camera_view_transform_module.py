"""Lift camera features and pool them into the LiDAR BEV frame."""

from __future__ import annotations

import torch
import torch.nn as nn

from common.bevfusion_depth_distribution_head import BEVFusionDepthDistributionHead
from fusion_unit.asymmetric.feature.ops.bevfusion_camera_bev_pooling import (
    bevfusion_camera_bev_pooling,
)
from utils.bevfusion_camera_geometry_transform import (
    bevfusion_camera_geometry_transform,
)
from utils.bevfusion_camera_lift_2d_to_3d import bevfusion_camera_lift_2d_to_3d
from utils.bevfusion_depth_weighted_feature_expand import (
    bevfusion_depth_weighted_feature_expand,
)


class CameraViewTransformModule(nn.Module):
    """Convert perspective image features into the LiDAR-coordinate BEV grid.

    Inputs:
        camera_features: ``(B, N, C, H_f, W_f)``.
        batch["calibs"]: per-sample KITTI ``P2``, ``R0_rect`` and
            ``Tr_velo2cam`` matrices. ``P2`` maps rectified camera coordinates
            to pixels; the inverse of ``R0_rect @ Tr_velo2cam`` maps camera
            coordinates into the LiDAR frame.

    Output:
        Camera BEV tensor ``(B, C, bev_h, bev_w)`` in LiDAR ``x-y`` coordinates.

    Operators:
        ``BEVFusionDepthDistributionHead``, ``bevfusion_camera_lift_2d_to_3d``,
        ``bevfusion_camera_geometry_transform``,
        ``bevfusion_depth_weighted_feature_expand`` and
        ``bevfusion_camera_bev_pooling``.
    """

    def __init__(
        self,
        channels: int,
        num_depth_bins: int,
        depth_min: float,
        depth_max: float,
        bev_h: int,
        bev_w: int,
        point_cloud_range,
    ):
        super().__init__()
        self.depth_head = BEVFusionDepthDistributionHead(
            channels, channels, num_depth_bins, use_bn=True
        )
        self.register_buffer("depths", torch.linspace(depth_min, depth_max, num_depth_bins))
        self.register_buffer("point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32))
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)

    @staticmethod
    def _to44(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape == (4, 4):
            return matrix
        output = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
        if matrix.shape == (3, 4):
            output[:3, :4] = matrix
        elif matrix.shape == (3, 3):
            output[:3, :3] = matrix
        else:
            raise ValueError("calibration matrix must be 3x3, 3x4 or 4x4")
        return output

    def _camera_to_lidar(self, calib: dict, feature: torch.Tensor):
        rect = self._to44(calib["R0_rect"].to(feature))
        velo_to_cam = self._to44(calib["Tr_velo2cam"].to(feature))
        transform = torch.linalg.inv(rect @ velo_to_cam)
        return transform[:3, :3], transform[:3, 3]

    def _intrinsic_and_pixels(self, calib: dict, feature: torch.Tensor, image_shape):
        feat_h, feat_w = feature.shape[-2:]
        image_h, image_w = image_shape
        p2 = calib["P2"].to(feature)
        intrinsic = p2[:3, :3].clone()
        intrinsic[0] *= float(feat_w) / float(image_w)
        intrinsic[1] *= float(feat_h) / float(image_h)
        ys, xs = torch.meshgrid(
            torch.arange(feat_h, device=feature.device, dtype=feature.dtype),
            torch.arange(feat_w, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        return intrinsic, torch.stack([xs, ys], dim=-1)

    def _indices(self, points_lidar: torch.Tensor) -> torch.Tensor:
        limits = self.point_cloud_range.to(points_lidar)
        x_index = torch.floor(
            (points_lidar[..., 0] - limits[0]) / (limits[3] - limits[0]) * self.bev_w
        ).long()
        y_index = torch.floor(
            (points_lidar[..., 1] - limits[1]) / (limits[4] - limits[1]) * self.bev_h
        ).long()
        valid = (
            (x_index >= 0)
            & (x_index < self.bev_w)
            & (y_index >= 0)
            & (y_index < self.bev_h)
            & (points_lidar[..., 2] >= limits[2])
            & (points_lidar[..., 2] < limits[5])
        )
        invalid = torch.full_like(x_index, -1)
        return torch.stack(
            [torch.where(valid, y_index, invalid), torch.where(valid, x_index, invalid)], dim=-1
        )

    def forward(self, camera_features: torch.Tensor, batch: dict) -> torch.Tensor:
        if camera_features.dim() != 5:
            raise ValueError("camera_features must have shape (B,N,C,H,W)")
        outputs = []
        image_shapes = batch.get("padded_image_shapes", batch.get("image_shapes"))
        for batch_index, cameras in enumerate(camera_features):
            camera_bevs = []
            for camera_feature in cameras:
                feature = camera_feature.unsqueeze(0)
                depth_prob = self.depth_head(feature)
                frustum_feature = bevfusion_depth_weighted_feature_expand(feature, depth_prob)
                image_shape = image_shapes[batch_index] if image_shapes else batch["images"].shape[-2:]
                intrinsic, pixels = self._intrinsic_and_pixels(
                    batch["calibs"][batch_index], camera_feature, image_shape
                )
                identity = torch.eye(3, device=feature.device, dtype=feature.dtype)
                zero = torch.zeros(3, device=feature.device, dtype=feature.dtype)
                points_camera = bevfusion_camera_lift_2d_to_3d(
                    pixels, self.depths.to(feature), intrinsic, identity, zero
                )
                rotation, translation = self._camera_to_lidar(
                    batch["calibs"][batch_index], camera_feature
                )
                points_lidar = bevfusion_camera_geometry_transform(
                    points_camera, rotation, translation
                )
                indices = self._indices(points_lidar).unsqueeze(0)
                camera_bevs.append(
                    bevfusion_camera_bev_pooling(
                        frustum_feature, indices, self.bev_h, self.bev_w
                    )
                )
            outputs.append(torch.stack(camera_bevs, dim=0).sum(dim=0))
        return torch.cat(outputs, dim=0)
