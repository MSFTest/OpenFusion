# -*- coding: utf-8 -*-
"""Faraway 与 near 检测结果合并模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FarawayMergeModule(nn.Module):
    """合并远距离 refinement 结果与 near detector 结果。

    Inputs:
        faraway_output: dict，包含 ``boxes_3d_camera (F,7)``、
            ``scores (F,)``、``labels (F,)``。
        near_output: dict，包含 ``boxes_3d_camera (N,7)``、
            ``scores (N,)``、``labels (N,)``。

    Returns:
        dict，包含拼接后的 ``boxes_3d_camera (M,7)``、``scores (M,)``、
        ``labels (M,)``。
    """

    def forward(
        self,
        faraway_output: Dict[str, torch.Tensor],
        near_output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        far_boxes = faraway_output["boxes_3d_camera"]
        near_boxes = near_output["boxes_3d_camera"].to(device=far_boxes.device, dtype=far_boxes.dtype)
        far_scores = faraway_output["scores"].to(device=far_boxes.device, dtype=far_boxes.dtype)
        near_scores = near_output["scores"].to(device=far_boxes.device, dtype=far_boxes.dtype)
        far_labels = faraway_output["labels"].to(device=far_boxes.device, dtype=torch.long)
        near_labels = near_output["labels"].to(device=far_boxes.device, dtype=torch.long)

        return {
            "boxes_3d_camera": torch.cat([far_boxes, near_boxes], dim=0),
            "scores": torch.cat([far_scores, near_scores], dim=0),
            "labels": torch.cat([far_labels, near_labels], dim=0),
            "is_faraway_prediction": torch.cat(
                [
                    torch.ones((far_boxes.shape[0],), dtype=torch.bool, device=far_boxes.device),
                    torch.zeros((near_boxes.shape[0],), dtype=torch.bool, device=far_boxes.device),
                ],
                dim=0,
            ),
        }
