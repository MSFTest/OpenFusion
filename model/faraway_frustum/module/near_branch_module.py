# -*- coding: utf-8 -*-
"""Near 目标检测结果整理模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class NearBranchModule(nn.Module):
    """整理冻结 3D detector 的 near 分支输出。

    Inputs:
        detector_output: dict，至少包含 ``boxes_3d_camera (N,7)``、
            ``scores (N,)``、``labels (N,)``。

    Returns:
        dict，字段同输入，空结果时返回形状正确的空张量。
    """

    def forward(self, detector_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        boxes = detector_output.get("boxes_3d_camera")
        scores = detector_output.get("scores")
        labels = detector_output.get("labels")
        if boxes is None:
            device = scores.device if scores is not None else torch.device("cpu")
            boxes = torch.zeros((0, 7), dtype=torch.float32, device=device)
        if scores is None:
            scores = boxes.new_zeros((boxes.shape[0],))
        if labels is None:
            labels = torch.zeros((boxes.shape[0],), dtype=torch.long, device=boxes.device)
        return {
            "boxes_3d_camera": boxes,
            "scores": scores,
            "labels": labels.long(),
        }
