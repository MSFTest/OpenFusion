# -*- coding: utf-8 -*-
"""Proposal-level fusion refinement module for EPNet."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ProposalFusionRefineModule(nn.Module):
    """
    用框内融合点特征细化冻结 detector 的 proposal。

    Inputs:
        detector_boxes: (M, 7)，LiDAR 坐标 [x, y, z, dx, dy, dz, yaw]。
        detector_scores: (M,)，冻结 detector 原始分数。
        detector_labels: (M,)，冻结 detector 类别索引。
        xyz: (B, N, 3)，LiDAR 坐标点。
        fused_features: (B, C, N)，融合点特征。
        point_cls_logits: (B, N)，点级前景 logits。

    Returns:
        dict:
            boxes_3d: (M, 7)，融合特征细化后的 LiDAR 框。
            scores_3d: (M,)，融合特征细化后的分数。
            labels_3d: (M,)，类别索引。
            proposal_score_logits: (M,)，proposal 级分类 logits。
            proposal_loc_conf: (M,)，proposal 级定位置信度。
            proposal_box_delta: (M, 7)，预测的归一化框残差。
            proposal_pooled_features: (M, C+2)，框内融合池化特征。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 256, max_center_shift: float = 0.2):
        super().__init__()
        self.max_center_shift = float(max_center_shift)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.score_delta = nn.Linear(hidden_channels, 1)
        self.loc_conf = nn.Linear(hidden_channels, 1)
        self.box_delta = nn.Linear(hidden_channels, 7)
        nn.init.zeros_(self.score_delta.weight)
        nn.init.zeros_(self.score_delta.bias)
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)
        nn.init.zeros_(self.loc_conf.weight)
        nn.init.zeros_(self.loc_conf.bias)

    @staticmethod
    def _inside_axis_aligned(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        centers = boxes[:, :3]
        half = boxes[:, 3:6].abs() / 2.0
        lower = centers - half
        upper = centers + half
        inside = (points[:, None, :] >= lower[None, :, :]) & (points[:, None, :] <= upper[None, :, :])
        return inside.all(dim=-1).transpose(0, 1).contiguous()

    @staticmethod
    def _score_to_logit(scores: torch.Tensor) -> torch.Tensor:
        scores = scores.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(scores / (1.0 - scores))

    def _pool_features(
        self,
        boxes: torch.Tensor,
        xyz: torch.Tensor,
        fused_features: torch.Tensor,
        point_cls_logits: torch.Tensor,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return fused_features.new_zeros((0, fused_features.shape[1] + 2))
        points = xyz[0]
        features = fused_features[0].transpose(0, 1).contiguous()
        point_prob = torch.sigmoid(point_cls_logits[0])
        inside = self._inside_axis_aligned(points, boxes)
        pooled = []
        for box, mask in zip(boxes, inside):
            if mask.any():
                feat = features[mask].mean(dim=0)
                prob = point_prob[mask].mean().view(1)
                occ = mask.float().mean().view(1)
            else:
                nearest = torch.cdist(box[None, :3], points).argmin(dim=1)[0]
                feat = features[nearest]
                prob = point_prob[nearest].view(1)
                occ = boxes.new_zeros(1)
            pooled.append(torch.cat([feat, prob.to(feat.dtype), occ.to(feat.dtype)], dim=0))
        return torch.stack(pooled, dim=0)

    def forward(
        self,
        detector_boxes: torch.Tensor,
        detector_scores: torch.Tensor,
        detector_labels: torch.Tensor,
        xyz: torch.Tensor,
        fused_features: torch.Tensor,
        point_cls_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if detector_scores.numel() == 0:
            return {
                "boxes_3d": detector_boxes,
                "scores_3d": detector_scores,
                "labels_3d": detector_labels,
                "proposal_score_logits": detector_scores.new_zeros((0,)),
                "proposal_loc_conf": detector_scores.new_zeros((0,)),
                "proposal_box_delta": detector_boxes.new_zeros((0, 7)),
                "proposal_pooled_features": fused_features.new_zeros((0, fused_features.shape[1] + 2)),
            }

        pooled = self._pool_features(detector_boxes, xyz, fused_features, point_cls_logits)
        hidden = self.mlp(pooled)
        raw_delta = self.box_delta(hidden)
        dims = detector_boxes[:, 3:6].abs().clamp(min=1e-3)
        refined_center = detector_boxes[:, :3] + torch.tanh(raw_delta[:, :3]) * dims * self.max_center_shift
        refined_size = detector_boxes[:, 3:6] * torch.exp(raw_delta[:, 3:6].clamp(min=-0.2, max=0.2))
        refined_yaw = detector_boxes[:, 6:7] + torch.tanh(raw_delta[:, 6:7]) * 0.2
        refined_boxes = torch.cat([refined_center, refined_size, refined_yaw], dim=1)
        score_logits = self._score_to_logit(detector_scores) + self.score_delta(hidden).squeeze(1)
        refined_scores = torch.sigmoid(score_logits)
        loc_conf = torch.sigmoid(self.loc_conf(hidden).squeeze(1))
        return {
            "boxes_3d": refined_boxes,
            "scores_3d": refined_scores,
            "labels_3d": detector_labels,
            "proposal_score_logits": score_logits,
            "proposal_loc_conf": loc_conf,
            "proposal_box_delta": raw_delta,
            "proposal_pooled_features": pooled,
        }
