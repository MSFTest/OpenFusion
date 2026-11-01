# -*- coding: utf-8 -*-
"""近距离 3D 检测与图像证据的分数融合模块。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _boxes2d_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0))[None, :]
    return inter / (area_a + area_b - inter).clamp_min(1e-6)


def _boxes3d_camera_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8, 3))
    x, y, z, w, l, h, ry = boxes.unbind(dim=1)
    x_c = torch.stack([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dim=1)
    y_c = torch.stack([torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h), -h, -h, -h, -h], dim=1)
    z_c = torch.stack([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dim=1)
    cos_r = torch.cos(ry)
    sin_r = torch.sin(ry)
    corner_x = cos_r[:, None] * x_c + sin_r[:, None] * z_c + x[:, None]
    corner_y = y_c + y[:, None]
    corner_z = -sin_r[:, None] * x_c + cos_r[:, None] * z_c + z[:, None]
    return torch.stack([corner_x, corner_y, corner_z], dim=2)


class NearImageScoreFusionModule(nn.Module):
    """用图像 2D 检测证据校准 near 3D detector 分数。

    Inputs:
        near_output: dict，包含 ``boxes_3d_camera (N,7)``、``scores (N,)``、
            ``labels (N,)``。3D box 为 KITTI camera rect 坐标，格式
            ``(x,y,z,w,l,h,ry)``。
        image_instances: dict，包含 ``boxes_2d (K,4)``、``scores (K,)``、
            ``labels (K,)``，图像坐标为像素 ``(x1,y1,x2,y2)``。
        p2: ``(3,4)`` KITTI camera P2 投影矩阵。
        image_shape: ``(H,W)``。

    Returns:
        dict，与 ``near_output`` 同结构，但 ``scores`` 根据同类别 2D IoU
        与图像分数进行融合校准。该模块只改变排序分数，不改 3D box 坐标。
    """

    def __init__(
        self,
        enabled: bool = False,
        power: float = 0.5,
        min_factor: float = 0.65,
        trainable: bool = False,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.power = float(power)
        self.min_factor = float(min_factor)
        self.trainable = bool(trainable)
        self.residual_scale = float(residual_scale)
        self.calibration_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.calibration_head[-1].weight)
        nn.init.zeros_(self.calibration_head[-1].bias)
        if not self.trainable:
            for parameter in self.calibration_head.parameters():
                parameter.requires_grad_(False)

    def _project_boxes_to_2d(self, boxes: torch.Tensor, p2: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        corners = _boxes3d_camera_to_corners(boxes)
        if corners.numel() == 0:
            return boxes.new_zeros((0, 4))
        ones = torch.ones((*corners.shape[:2], 1), dtype=corners.dtype, device=corners.device)
        corners_h = torch.cat([corners, ones], dim=2)
        proj = torch.matmul(corners_h, p2.to(device=boxes.device, dtype=boxes.dtype).t())
        uv = proj[..., :2] / proj[..., 2:3].clamp_min(1e-6)
        x1 = uv[..., 0].amin(dim=1).clamp(0, float(image_shape[1] - 1))
        y1 = uv[..., 1].amin(dim=1).clamp(0, float(image_shape[0] - 1))
        x2 = uv[..., 0].amax(dim=1).clamp(0, float(image_shape[1] - 1))
        y2 = uv[..., 1].amax(dim=1).clamp(0, float(image_shape[0] - 1))
        return torch.stack([x1, y1, x2, y2], dim=1)

    def forward(
        self,
        near_output: Dict[str, torch.Tensor],
        image_instances: Dict[str, torch.Tensor],
        p2: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if not self.enabled or near_output["boxes_3d_camera"].numel() == 0:
            return near_output
        raw_scores = near_output["scores"]
        image_boxes = image_instances["boxes_2d"].to(device=raw_scores.device, dtype=raw_scores.dtype)
        image_scores = image_instances["scores"].to(device=raw_scores.device, dtype=raw_scores.dtype)
        image_labels = image_instances["labels"].to(device=near_output["labels"].device, dtype=torch.long)
        near_boxes_2d = self._project_boxes_to_2d(near_output["boxes_3d_camera"], p2, image_shape)
        if image_boxes.numel() == 0:
            max_support = raw_scores.new_zeros(raw_scores.shape)
            max_iou = raw_scores.new_zeros(raw_scores.shape)
            matched_image_scores = raw_scores.new_zeros(raw_scores.shape)
        else:
            ious = _boxes2d_iou(near_boxes_2d, image_boxes)
            same_class = near_output["labels"][:, None].to(dtype=torch.long) == image_labels[None, :]
            class_ious = ious * same_class.to(dtype=ious.dtype)
            support = class_ious * image_scores[None, :]
            max_support, support_indices = support.max(dim=1)
            max_support = max_support.clamp(0.0, 1.0)
            max_iou = class_ious.amax(dim=1).clamp(0.0, 1.0)
            matched_image_scores = image_scores[support_indices]
            has_support = same_class.any(dim=1)
            matched_image_scores = torch.where(
                has_support,
                matched_image_scores,
                torch.zeros_like(matched_image_scores),
            )
        factor = self.min_factor + (1.0 - self.min_factor) * max_support.pow(self.power)
        base_scores = (raw_scores * factor).clamp(1e-4, 1.0 - 1e-4)
        projected_area = (
            (near_boxes_2d[:, 2] - near_boxes_2d[:, 0]).clamp_min(0.0)
            * (near_boxes_2d[:, 3] - near_boxes_2d[:, 1]).clamp_min(0.0)
        )
        image_area = max(float(image_shape[0] * image_shape[1]), 1.0)
        features = torch.stack(
            [
                raw_scores.clamp(0.0, 1.0),
                max_support,
                max_iou,
                matched_image_scores.clamp(0.0, 1.0),
                (near_output["boxes_3d_camera"][:, 2] / 80.0).clamp(0.0, 2.0),
                torch.sqrt((projected_area / image_area).clamp_min(0.0)).clamp(0.0, 1.0),
            ],
            dim=1,
        ).detach()
        if self.trainable:
            calibration_residual = self.residual_scale * torch.tanh(
                self.calibration_head(features).squeeze(1)
            )
        else:
            calibration_residual = base_scores.new_zeros(base_scores.shape)
        calibration_logits = torch.logit(base_scores) + calibration_residual
        fused = dict(near_output)
        fused["scores"] = torch.sigmoid(calibration_logits)
        fused["raw_scores"] = raw_scores
        fused["base_scores"] = base_scores
        fused["image_support_scores"] = max_support
        fused["image_iou_scores"] = max_iou
        fused["matched_image_scores"] = matched_image_scores
        fused["calibration_features"] = features
        fused["calibration_residual"] = calibration_residual
        fused["calibration_logits"] = calibration_logits
        return fused
