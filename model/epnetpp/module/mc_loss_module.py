# -*- coding: utf-8 -*-
"""EPNet++ Multi-Modal Consistency loss module."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils  # noqa: F401 - registers fusionrepo operators


class MCLossModule(nn.Module):
    """
    计算点云 confidence 与图像 confidence 的一致性损失。

    Args:
        point_logits: Tensor，shape 为 (B, N)。
        image_conf_map: Tensor，shape 为 (B, 1, H, W)。
        feature_xy_norm: Tensor，shape 为 (B, N, 2)，对应 image_conf_map 的归一化坐标。

    Returns:
        dict，包含 loss_mc、loss_i2p、loss_p2i、image_point_conf。
    """

    def __init__(self, threshold: float = 0.2, lambda_i2p: float = 0.5, lambda_p2i: float = 0.5):
        super().__init__()
        self.threshold = float(threshold)
        self.lambda_i2p = float(lambda_i2p)
        self.lambda_p2i = float(lambda_p2i)

    def forward(
        self,
        point_logits: torch.Tensor,
        image_conf_map: torch.Tensor,
        feature_xy_norm: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_point_logits = torch.ops.fusionrepo.epnetpp_image_point_sampling(
            image_conf_map,
            feature_xy_norm,
        ).squeeze(1)
        point_prob = torch.sigmoid(point_logits)
        image_prob = torch.sigmoid(image_point_logits)
        avg_prob = ((point_prob + image_prob) * 0.5).detach().clamp(1e-4, 1.0 - 1e-4)
        mask = (torch.maximum(point_prob.detach(), image_prob.detach()) >= self.threshold).float()
        loss_i2p = F.binary_cross_entropy(image_prob.clamp(1e-4, 1.0 - 1e-4), avg_prob, reduction="none")
        loss_p2i = F.binary_cross_entropy(point_prob.clamp(1e-4, 1.0 - 1e-4), avg_prob, reduction="none")
        denom = mask.sum().clamp_min(1.0)
        loss_i2p = (loss_i2p * mask).sum() / denom
        loss_p2i = (loss_p2i * mask).sum() / denom
        return {
            "loss_mc": self.lambda_i2p * loss_i2p + self.lambda_p2i * loss_p2i,
            "loss_i2p": loss_i2p.detach(),
            "loss_p2i": loss_p2i.detach(),
            "image_point_conf": image_prob.detach(),
        }
