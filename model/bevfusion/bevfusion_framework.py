"""BEVFusion framework containing orchestration only."""

from __future__ import annotations

import torch.nn as nn


class BEVFusionFramework(nn.Module):
    """Schedule BEVFusion modules without implementing geometry or fusion logic."""

    def __init__(
        self,
        camera_feature_module: nn.Module,
        camera_view_transform_module: nn.Module,
        lidar_bev_module: nn.Module,
        bev_fusion_module: nn.Module,
        detection_task_module: nn.Module,
    ):
        super().__init__()
        self.camera_feature_module = camera_feature_module
        self.camera_view_transform_module = camera_view_transform_module
        self.lidar_bev_module = lidar_bev_module
        self.bev_fusion_module = bev_fusion_module
        self.detection_task_module = detection_task_module

    def _dispatch(self, batch: dict, mode: str):
        camera_features = self.camera_feature_module(batch)
        camera_bev = self.camera_view_transform_module(camera_features, batch)
        lidar_bev = self.lidar_bev_module(batch)
        fused_bev = self.bev_fusion_module(camera_bev, lidar_bev)
        return self.detection_task_module(fused_bev, batch, mode=mode)

    def train_forward(self, batch: dict):
        return self._dispatch(batch, mode="train")

    def test_forward(self, batch: dict):
        return self._dispatch(batch, mode="test")

    def forward(self, batch: dict, mode: str | None = None):
        selected = mode or ("train" if self.training else "test")
        if selected == "train":
            return self.train_forward(batch)
        if selected == "test":
            return self.test_forward(batch)
        raise ValueError("mode must be 'train' or 'test'")
