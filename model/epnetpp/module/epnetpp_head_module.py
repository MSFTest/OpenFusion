# -*- coding: utf-8 -*-
"""EPNet++ trainable point head and score refinement module."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class EPNetPPHeadModule(nn.Module):
    """
    基于 CB-Fusion 点特征预测训练目标，并对 detector score 做融合重加权。

    Args:
        point_features: Tensor，shape 为 (B, C, N)。
        detector_prediction: dict，含 boxes_3d/scores_3d/labels_3d。

    Returns:
        dict:
            point_cls_logits: (B, N)
            point_box_delta: (B, N, 7)
            point_loc_conf: (B, N)
            prediction: 融合修正后的 boxes/scores/labels，score 仍源自真实 detector。
    """

    def __init__(self, in_channels: int, max_score_logit_delta: float = 0.25):
        super().__init__()
        self.max_score_logit_delta = float(max_score_logit_delta)
        hidden = max(64, int(in_channels))
        self.shared = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv1d(hidden, 1, 1)
        self.box = nn.Conv1d(hidden, 7, 1)
        self.loc = nn.Conv1d(hidden, 1, 1)
        self.score_delta = nn.Conv1d(hidden, 1, 1)
        nn.init.zeros_(self.score_delta.weight)
        nn.init.zeros_(self.score_delta.bias)

    @staticmethod
    def _points_in_axis_aligned_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        xyz = points[:, :3]
        centers = boxes[:, :3]
        half = boxes[:, 3:6].abs() / 2.0
        lower = centers - half
        upper = centers + half
        inside = (xyz[:, None, :] >= lower[None, :, :]) & (xyz[:, None, :] <= upper[None, :, :])
        return inside.all(dim=-1)

    def _box_reduce(self, point_values: torch.Tensor, points: Optional[torch.Tensor], boxes: torch.Tensor):
        if boxes.numel() == 0:
            return boxes.new_zeros((0,)), boxes.new_zeros((0,), dtype=torch.bool)
        if points is None:
            base = point_values.mean(dim=1).reshape(-1)[0].to(device=boxes.device, dtype=boxes.dtype)
            return base.expand(boxes.shape[0]), torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
        point_xyz = points[0, :, :3].to(device=boxes.device, dtype=boxes.dtype)
        values = point_values[0].to(device=boxes.device, dtype=boxes.dtype)
        inside = self._points_in_axis_aligned_boxes(point_xyz, boxes)
        counts = inside.sum(dim=0)
        valid = counts > 0
        summed = (inside.to(values.dtype) * values[:, None]).sum(dim=0)
        fallback = values.mean().expand_as(summed)
        reduced = torch.where(valid, summed / counts.clamp_min(1).to(values.dtype), fallback)
        return reduced, valid

    def forward(
        self,
        point_features: torch.Tensor,
        detector_prediction: Dict[str, torch.Tensor],
        points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.shared(point_features)
        cls_logits = self.cls(feat).squeeze(1)
        loc_conf = torch.sigmoid(self.loc(feat).squeeze(1))
        point_score_delta = self.score_delta(feat).squeeze(1)
        delta = self.box(feat).transpose(1, 2).contiguous()

        boxes = detector_prediction["boxes_3d"]
        scores = detector_prediction["scores_3d"]
        labels = detector_prediction["labels_3d"]
        box_loc_conf, box_conf_valid = self._box_reduce(loc_conf, points, boxes)
        box_delta_logit, _ = self._box_reduce(point_score_delta, points, boxes)
        raw_scores = scores
        score_refine_prob = scores
        if scores.numel() > 0:
            raw_score_safe = raw_scores.to(device=scores.device, dtype=scores.dtype).clamp(1e-4, 1.0 - 1e-4)
            raw_logit = torch.logit(raw_score_safe)
            bounded_delta = torch.tanh(box_delta_logit.to(device=scores.device, dtype=scores.dtype))
            bounded_delta = bounded_delta * self.max_score_logit_delta
            scores = torch.sigmoid(raw_logit + bounded_delta).clamp(0.0, 1.0)
            score_refine_prob = scores
        return {
            "point_cls_logits": cls_logits,
            "point_box_delta": delta,
            "point_loc_conf": loc_conf,
            "score_refine_conf": score_refine_prob,
            "score_refine_delta_logit": box_delta_logit,
            "score_refine_loc_conf": box_loc_conf.clamp(1e-4, 1.0 - 1e-4),
            "score_refine_valid": box_conf_valid,
            "raw_scores_3d": raw_scores,
            "raw_prediction": {
                "boxes_3d": boxes,
                "scores_3d": raw_scores,
                "labels_3d": labels,
            },
            "prediction": {
                "boxes_3d": boxes,
                "scores_3d": scores,
                "labels_3d": labels,
            },
        }
