# -*- coding: utf-8 -*-
"""EPNet++ cascade bi-directional fusion module."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .feature_alignment_module import FeatureAlignmentModule
from .geometry_projection_module import GeometryProjectionModule
from .il_fusion_module import ILFusionModule
from .li_fusion_module import LIFusionModule


class CBFusionModule(nn.Module):
    """
    EPNet++ 级联双向融合模块：先 P2I 的 IL-Fusion，再 I2P 的 LI-Fusion。

    Args:
        points: Tensor，shape 为 (B, N, C_point)，LiDAR 坐标/点特征。
        point_features: Tensor，shape 为 (B, C_p, N)。
        image_grid_features: Tensor，shape 为 (B, C_i, H, W)。
        calib tensors: LiDAR 到 image 的投影参数。

    Returns:
        dict:
            cb_point_features: (B, C_fuse, N)
            cb_image_features: (B, C_fuse, H, W)
            i2p_weight: (B, 1, N)
            p2i_weight: (B, 1, N)
    """

    def __init__(self, point_in_channels: int, image_in_channels: int, fusion_channels: int):
        super().__init__()
        self.geometry = GeometryProjectionModule()
        self.align = FeatureAlignmentModule(point_in_channels, image_in_channels, fusion_channels)
        self.il_fusion = ILFusionModule(fusion_channels)
        self.li_fusion = LIFusionModule(fusion_channels)

    @staticmethod
    def _grid_indices(uv: torch.Tensor, image_hw: Tuple[int, int], feature_hw: Tuple[int, int]) -> torch.Tensor:
        image_h, image_w = image_hw
        feat_h, feat_w = feature_hw
        x = torch.floor(uv[..., 0] * float(feat_w) / max(float(image_w), 1.0)).long()
        y = torch.floor(uv[..., 1] * float(feat_h) / max(float(image_h), 1.0)).long()
        return torch.stack([y, x], dim=-1)

    def forward(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        image_grid_features: torch.Tensor,
        lidar_to_cam_r: torch.Tensor,
        lidar_to_cam_t: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        geom = self.geometry(
            points=points,
            image_features=image_grid_features,
            lidar_to_cam_r=lidar_to_cam_r,
            lidar_to_cam_t=lidar_to_cam_t,
            cam_intrinsic=cam_intrinsic,
            image_hw=image_hw,
        )
        aligned = self.align(point_features, geom["image_point_features"])
        grid_idx = self._grid_indices(geom["uv"], image_hw, tuple(image_grid_features.shape[-2:]))
        il_out = self.il_fusion(
            aligned["point_aligned"],
            aligned["image_aligned"],
            image_grid_features,
            grid_idx,
        )
        geom_after_il = self.geometry(
            points=points,
            image_features=il_out["enhanced_image_features"],
            lidar_to_cam_r=lidar_to_cam_r,
            lidar_to_cam_t=lidar_to_cam_t,
            cam_intrinsic=cam_intrinsic,
            image_hw=image_hw,
        )
        li_out = self.li_fusion(aligned["point_aligned"], geom_after_il["image_point_features"])
        return {
            **geom,
            "point_aligned": aligned["point_aligned"],
            "image_aligned": aligned["image_aligned"],
            "cb_point_features": li_out["enhanced_point_features"],
            "cb_image_features": il_out["enhanced_image_features"],
            "i2p_weight": li_out["i2p_weight"],
            "p2i_weight": il_out["p2i_weight"],
            "point_grid_idx": grid_idx,
            "scattered_point_features": il_out["scattered_point_features"],
        }
