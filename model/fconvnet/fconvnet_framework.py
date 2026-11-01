# -*- coding: utf-8 -*-
"""
FConvNetFramework

说明：
    F-ConvNet 框架调度层。Framework 只负责调用 2D 检测器接口、
    点云 Backbone 接口和各个 module，不直接实现融合算子逻辑。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from model.fconvnet.module import (
    FConvNetDetectionHead,
    FConvNetRefinementModule,
    FrustumFCNModule,
    FrustumGenerationModule,
    FrustumPointNetEncodingModule,
    SequenceFeatureModule,
)


class FConvNetFramework(nn.Module):
    """
    F-ConvNet 网络框架。

    Args:
        detector2d_backbone: 2D 检测器接口，输入 batch 后输出 boxes_2d/scores_2d/labels_2d。
        point_cloud_backbone: 点云 Backbone 接口，输入 batch 后输出 points/cam_intrinsic 等。
    """

    def __init__(
        self,
        detector2d_backbone: nn.Module,
        point_cloud_backbone: nn.Module,
        bin_size: float = 0.25,
        stride: float = 0.25,
        max_points_per_frustum: int = 128,
        max_depth: float = 70.0,
        point_channels: tuple[int, ...] = (64, 128, 256),
        sequence_channels: int = 128,
        num_classes_2d: int = 3,
        num_heading_bins: int = 12,
        num_size_clusters: int = 1,
        enable_refinement: bool = False,
        train_refinement_only: bool = False,
        refine_residual_scale: float = 1.0,
        refine_residual_gate_score: float = float("inf"),
    ):
        super().__init__()
        self.detector2d_backbone = detector2d_backbone
        self.point_cloud_backbone = point_cloud_backbone
        self.frustum_generation = FrustumGenerationModule(
            bin_size=bin_size,
            stride=stride,
            max_points_per_frustum=max_points_per_frustum,
            max_depth=max_depth,
        )
        self.frustum_pointnet = FrustumPointNetEncodingModule(
            in_channels=3,
            channels=point_channels,
        )
        self.frustum_fcn = FrustumFCNModule(in_channels=128, num_classes=num_classes_2d)
        self.detection_head = FConvNetDetectionHead(
            in_channels=self.frustum_fcn.out_channels,
            num_heading_bins=num_heading_bins,
            num_size_clusters=num_size_clusters,
        )
        self.refinement = FConvNetRefinementModule(
            max_points_per_group=max_points_per_frustum,
            residual_scale=refine_residual_scale,
            residual_gate_score=refine_residual_gate_score,
        ) if enable_refinement else None
        self.train_refinement_only = bool(train_refinement_only)
        if self.train_refinement_only:
            for module in (self.frustum_pointnet, self.frustum_fcn, self.detection_head):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    @staticmethod
    def _align_center_refs(center_refs: torch.Tensor, output_length: int) -> torch.Tensor:
        return center_refs[:, ::2, :][:, :output_length, :]

    @staticmethod
    def _boxes_from_center_view(boxes: torch.Tensor, rot_angles: torch.Tensor) -> torch.Tensor:
        angles = rot_angles.to(boxes).view(-1, 1)
        c, s = torch.cos(angles), torch.sin(angles)
        out = boxes.clone()
        x, z = boxes[..., 0], boxes[..., 2]
        out[..., 0] = c * x + s * z
        out[..., 2] = -s * x + c * z
        out[..., 6] = torch.remainder(boxes[..., 6] + angles + torch.pi, 2 * torch.pi) - torch.pi
        return out

    def _stage1_forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        lidar_out = self.point_cloud_backbone(batch)
        det2d_out = self.detector2d_backbone(batch)

        frustum_out = self.frustum_generation(
            points=lidar_out["points"],
            det2d=det2d_out,
            cam_intrinsic=lidar_out["cam_intrinsic"],
            image_shape=batch.get("image_shape"),
        )
        frustum_feats = self.frustum_pointnet(
            frustum_out["multiscale_points"],
            frustum_out["multiscale_masks"],
            frustum_out["multiscale_center_refs"],
        )
        fused_features = self.frustum_fcn(frustum_feats, frustum_out["proposal_labels"])
        aligned_refs = frustum_out["multiscale_center_refs"][1][:, :fused_features.shape[-1]]
        pred = self.detection_head(fused_features, aligned_refs)
        pred["boxes_3d_center_view"] = pred["boxes_3d"]
        pred["boxes_3d"] = self._boxes_from_center_view(
            pred["boxes_3d"], frustum_out["frustum_rot_angles"]
        )
        pred.update(frustum_out)
        pred["center_refs"] = aligned_refs
        pred["lidar_out"] = lidar_out
        return pred

    def _forward_impl(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.train_refinement_only:
            for module in (self.frustum_pointnet, self.frustum_fcn, self.detection_head):
                module.eval()
            with torch.no_grad():
                stage1 = self._stage1_forward(batch)
        else:
            stage1 = self._stage1_forward(batch)
        if self.refinement is None:
            return stage1
        return self.refinement(stage1, stage1["lidar_out"]["points"], batch.get("gt_boxes_3d"))

    def train_forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._forward_impl(batch)

    @torch.no_grad()
    def test_forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._forward_impl(batch)
