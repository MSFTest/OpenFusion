# -*- coding: utf-8 -*-
"""
FConvNetDetectionHead

功能：
    对 FCN 融合后的 frustum 序列特征执行分类与 3D box 回归预测。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from common.shared_mlp_1d import SharedMLP1D


class FConvNetDetectionHead(nn.Module):
    """
    F-ConvNet 检测头。

    Args:
        in_channels: 输入通道数。
        hidden_channels: head 隐藏通道数。
        num_heading_bins: heading 分类 bin 数。
        num_size_clusters: size 类别数。

    Inputs:
        fused_features: (P, C, T)
        center_refs: (P, T, 3)

    Returns:
        dict，包含分类、回归原始预测以及粗解码 boxes。
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_heading_bins: int = 12,
        num_size_clusters: int = 1,
        mean_sizes: tuple[tuple[float, float, float], ...] = ((1.6, 3.9, 1.56),),
    ):
        super().__init__()
        self.num_heading_bins = int(num_heading_bins)
        self.num_size_clusters = int(num_size_clusters)
        self.register_buffer("mean_sizes", torch.tensor(mean_sizes, dtype=torch.float32))
        self.cls_out = nn.Conv1d(in_channels, 2, kernel_size=1)
        reg_channels = 3 + self.num_heading_bins * 2 + self.num_size_clusters + self.num_size_clusters * 3
        self.reg_out = nn.Conv1d(in_channels, reg_channels, kernel_size=1)

    def _decode_boxes(
        self,
        reg: torch.Tensor,
        center_refs: torch.Tensor,
    ) -> torch.Tensor:
        P, T, _ = center_refs.shape
        center_offset = reg[:, :3].transpose(1, 2).contiguous()
        heading_scores = reg[:, 3:3 + self.num_heading_bins].transpose(1, 2).contiguous()
        heading_res = reg[:, 3 + self.num_heading_bins:3 + self.num_heading_bins * 2].transpose(1, 2).contiguous()
        size_start = 3 + self.num_heading_bins * 2
        size_scores = reg[:, size_start:size_start + self.num_size_clusters].transpose(1, 2).contiguous()
        size_res = reg[:, size_start + self.num_size_clusters:].transpose(1, 2).contiguous()
        size_res = size_res.view(P, T, self.num_size_clusters, 3)

        center = center_refs + center_offset
        heading_cls = heading_scores.argmax(dim=-1)
        heading_res_sel = torch.gather(heading_res, 2, heading_cls.unsqueeze(-1)).squeeze(-1)
        angle_per_bin = 2.0 * torch.pi / float(self.num_heading_bins)
        heading = heading_cls.to(center.dtype) * angle_per_bin + heading_res_sel * (angle_per_bin / 2.0)
        heading = torch.remainder(heading + torch.pi, 2.0 * torch.pi) - torch.pi

        size_cls = size_scores.argmax(dim=-1)
        mean_sizes = self.mean_sizes.to(device=center.device, dtype=center.dtype)
        base_size = mean_sizes[size_cls]
        size_res_sel = torch.gather(
            size_res,
            2,
            size_cls.view(P, T, 1, 1).expand(P, T, 1, 3),
        ).squeeze(2)
        size = (base_size * (1.0 + size_res_sel)).clamp_min(1e-3)
        return torch.cat([center, size, heading.unsqueeze(-1)], dim=-1)

    def forward(self, fused_features: torch.Tensor, center_refs: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_logits = self.cls_out(fused_features).transpose(1, 2).contiguous()
        reg = self.reg_out(fused_features)
        boxes_3d = self._decode_boxes(reg, center_refs[:, :reg.shape[-1], :])
        return {
            "cls_logits": cls_logits,
            "reg_raw": reg.transpose(1, 2).contiguous(),
            "boxes_3d": boxes_3d,
            "center_refs": center_refs[:, :reg.shape[-1], :],
        }
