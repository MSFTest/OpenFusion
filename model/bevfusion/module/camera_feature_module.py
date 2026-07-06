"""Camera feature extraction and FPN composition for BEVFusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from common.bevfusion_camera_fpn_fusion import BEVFusionCameraFPNFusion


class CameraFeatureModule(nn.Module):
    """Extract and align camera features.

    Input:
        batch["images"]: ``(B, 3, H, W)`` or ``(B, N, 3, H, W)`` RGB tensor.

    Output:
        Camera feature tensor ``(B, N, C_out, H_f, W_f)``. KITTI uses ``N=1``.

    Operators:
        :class:`BEVFusionCameraFPNFusion`.
    """

    def __init__(self, image_adapter: nn.Module, in_channels, out_channels: int):
        super().__init__()
        self.image_adapter = image_adapter
        self.fpn_fusion = BEVFusionCameraFPNFusion(in_channels, out_channels)

    def forward(self, batch: dict) -> torch.Tensor:
        images = batch["images"]
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError("images must have shape (B,3,H,W) or (B,N,3,H,W)")
        batch_size, num_cameras = images.shape[:2]
        flat = images.reshape(-1, *images.shape[2:])
        features = self.image_adapter(flat)
        if not isinstance(features, (tuple, list)):
            features = [features]
        fused = self.fpn_fusion(features)
        return fused.reshape(batch_size, num_cameras, *fused.shape[1:])
