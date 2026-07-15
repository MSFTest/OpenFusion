# -*- coding: utf-8 -*-
"""EPNet framework."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model.epnet.module import (
    EPNetDetectionRefineModule,
    FeatureAlignmentModule,
    GeometryProjectionModule,
    LIFusionModule,
    OriginalStyleImageFusionModule,
    PointFusionHeadModule,
    ProposalFusionRefineModule,
)


class EPNetFramework(nn.Module):
    """
    EPNet 调度框架。

    Args:
        image_backbone: 图像特征接口，输入 images (B,3,H,W)，输出 (B,C_i,Hf,Wf)。
        point_cloud_detector: 点云检测接口，返回真实 detector 的 boxes/scores/labels。
        point_in_channels: 点特征输入通道数。
        image_in_channels: 图像点特征通道数。
        fusion_channels: 融合通道数。

    Notes:
        Framework 只调用接口与 module；几何投影、LI-Fusion、检测修正均在 module 内完成。
    """

    def __init__(
        self,
        image_backbone: nn.Module,
        point_cloud_detector: nn.Module,
        point_in_channels: int = 3,
        image_in_channels: int = 256,
        fusion_channels: int = 128,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.point_cloud_detector = point_cloud_detector
        self.geometry_projection = GeometryProjectionModule()
        self.feature_alignment = FeatureAlignmentModule(
            point_in_channels=point_in_channels,
            image_in_channels=image_in_channels,
            out_channels=fusion_channels,
        )
        self.li_fusion = LIFusionModule(fusion_channels)
        self.original_style_image_fusion = OriginalStyleImageFusionModule(
            point_channels=fusion_channels
        )
        self.point_head = PointFusionHeadModule(fusion_channels)
        self.detection_refine = ProposalFusionRefineModule(fusion_channels)

    def _forward_impl(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        points = batch["points"]
        if points.dim() == 2:
            points = points.unsqueeze(0)

        image_features = self.image_backbone(images)
        detector_out = self.point_cloud_detector(batch)
        geom = self.geometry_projection(
            points=points,
            image_features=image_features,
            lidar_to_cam_r=batch["lidar_to_cam_r"],
            lidar_to_cam_t=batch["lidar_to_cam_t"],
            cam_intrinsic=batch["cam_intrinsic"],
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
        )
        point_features = points[..., :3].transpose(1, 2).contiguous()
        aligned = self.feature_alignment(point_features, geom["image_point_features"])
        fused = self.li_fusion(aligned["point_aligned"], aligned["image_aligned"])
        original_style = self.original_style_image_fusion(
            images=images,
            point_features=fused["fused_features"],
            uv=geom["uv"],
            valid_mask=geom["valid_mask"],
        )
        head = self.point_head(original_style["original_style_features"])
        refined = self.detection_refine(
            detector_boxes=detector_out["boxes_3d"],
            detector_scores=detector_out["scores_3d"],
            detector_labels=detector_out["labels_3d"],
            xyz=geom["xyz"],
            fused_features=original_style["original_style_features"],
            point_cls_logits=head["point_cls_logits"],
        )
        return {
            **geom,
            **aligned,
            **fused,
            **original_style,
            **head,
            "detector_prediction": detector_out,
            "prediction": refined,
        }

    def train_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """训练前向，仅调度接口与 module。"""
        return self._forward_impl(batch)

    def test_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """推理前向，仅调度接口与 module。"""
        return self._forward_impl(batch)

    def forward(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> Dict[str, torch.Tensor]:
        if mode == "train":
            return self.train_forward(batch)
        if mode == "test":
            return self.test_forward(batch)
        raise ValueError(f"Unsupported EPNet mode: {mode}")
