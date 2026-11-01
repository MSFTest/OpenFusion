# -*- coding: utf-8 -*-
"""EPNet++ framework scheduling layer."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model.epnetpp.module import CBFusionModule, EPNetPPHeadModule, ImageStreamModule, PointRPNAdapterModule


class EPNetPPFramework(nn.Module):
    """
    EPNet++ 调度框架。

    Framework 只负责调用 backbone adapter 与 module，不直接写投影、融合或后处理逻辑。

    Args:
        image_backbone: 图像特征接口，输入 images (B,3,H,W)，输出 (B,C,Hf,Wf)。
        point_cloud_detector: 真实 mmdet3d detector adapter。
        image_in_channels: 图像 backbone 输出通道。
        point_in_channels: 点级输入通道。
        fusion_channels: 融合通道。
    """

    def __init__(
        self,
        image_backbone: nn.Module,
        point_cloud_detector: nn.Module,
        image_in_channels: int = 256,
        point_in_channels: int = 3,
        fusion_channels: int = 128,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.point_adapter = PointRPNAdapterModule(point_cloud_detector)
        self.image_stream = ImageStreamModule(image_in_channels, fusion_channels)
        self.cb_fusion = CBFusionModule(point_in_channels, fusion_channels, fusion_channels)
        self.head = EPNetPPHeadModule(fusion_channels)

    def _forward_impl(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        points = batch["points"]
        if points.dim() == 2:
            points = points.unsqueeze(0)

        image_features = self.image_backbone(images)
        image_out = self.image_stream(image_features)
        point_out = self.point_adapter(batch)
        cb_out = self.cb_fusion(
            points=point_out["points_batched"],
            point_features=point_out["point_features"],
            image_grid_features=image_out["image_grid_features"],
            lidar_to_cam_r=batch["lidar_to_cam_r"],
            lidar_to_cam_t=batch["lidar_to_cam_t"],
            cam_intrinsic=batch["cam_intrinsic"],
            image_hw=(int(images.shape[-2]), int(images.shape[-1])),
        )
        head_out = self.head(cb_out["cb_point_features"], point_out["detector_prediction"], point_out["points_batched"])
        return {
            **image_out,
            **point_out,
            **cb_out,
            **head_out,
        }

    def train_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """训练前向，只调度 module。"""
        return self._forward_impl(batch)

    def test_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """推理前向，只调度 module。"""
        return self._forward_impl(batch)

    def forward(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> Dict[str, torch.Tensor]:
        if mode == "train":
            return self.train_forward(batch)
        if mode == "test":
            return self.test_forward(batch)
        raise ValueError(f"Unsupported EPNet++ mode: {mode}")
