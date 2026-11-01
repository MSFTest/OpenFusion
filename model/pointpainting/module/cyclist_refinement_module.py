# -*- coding: utf-8 -*-
"""
CyclistRefinementModule

PointPainting cyclist/rider 语义一致性修正模块。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import utils  # noqa: F401


class CyclistRefinementModule(nn.Module):
    """
    对 painted points 中 cyclist/rider 语义通道做阈值修正。

    输入:
        painted_points: (N, D + C)

    输出:
        refined_points: (N, D + C)
    """

    def __init__(self, cyclist_index: int = -1, threshold: float = 0.3):
        super().__init__()
        self.cyclist_index = int(cyclist_index)
        self.threshold = float(threshold)

    def forward(self, painted_points: torch.Tensor) -> torch.Tensor:
        return torch.ops.fusionrepo.pointpainting_cyclist_semantic_refinement(
            painted_points,
            self.cyclist_index,
            self.threshold,
        )
