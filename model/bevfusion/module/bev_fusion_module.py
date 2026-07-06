"""Camera and LiDAR BEV fusion module."""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion_unit.symmetric.feature.block.bevfusion_encoder import BEVFusionEncoder
from fusion_unit.symmetric.feature.ops.bevfusion_bev_concat_fusion import (
    bevfusion_bev_concat_fusion,
)


class BEVFusionModule(nn.Module):
    """Concatenate and encode aligned camera/LiDAR BEV features.

    Inputs:
        camera_bev: ``(B, C_cam, H, W)`` in LiDAR BEV coordinates.
        lidar_bev: ``(B, C_lidar, H, W)`` in LiDAR BEV coordinates.

    Output:
        Fused BEV tensor ``(B, C_out, H, W)``.

    Operators:
        ``bevfusion_bev_concat_fusion`` and ``BEVFusionEncoder``.
    """

    def __init__(self, camera_channels: int, lidar_channels: int, out_channels: int):
        super().__init__()
        self.encoder = BEVFusionEncoder(
            camera_channels + lidar_channels, out_channels, out_channels, use_residual=False
        )

    def forward(self, camera_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        concatenated = bevfusion_bev_concat_fusion(camera_bev, lidar_bev, 1)
        return self.encoder(concatenated)
