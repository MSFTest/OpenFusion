"""Real detector execution and fusion-conditioned prediction refinement."""

from __future__ import annotations

import torch
import torch.nn as nn


class DetectionTaskModule(nn.Module):
    """Run a real mmdet3d detector and inject fused BEV context.

    Inputs:
        fused_bev: ``(B, C, H, W)``.
        batch["point_paths"]: paths consumed by the real mmdet3d detector.

    Output:
        Dictionary containing detector predictions. Each prediction stores
        LiDAR boxes ``(K, 7)``, detector-derived scores ``(K,)`` and class
        labels ``(K,)``. Trainable class-wise deltas are conditioned on the
        fused BEV tensor, so image/LiDAR fusion is not discarded.
    """

    def __init__(self, detector_adapter: nn.Module, fusion_channels: int, num_classes: int):
        super().__init__()
        self.detector_adapter = detector_adapter
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(fusion_channels, fusion_channels), nn.ReLU()
        )
        self.box_delta = nn.Linear(fusion_channels, num_classes * 7)
        self.score_delta = nn.Linear(fusion_channels, num_classes)
        self.num_classes = int(num_classes)
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)
        nn.init.zeros_(self.score_delta.weight)
        nn.init.zeros_(self.score_delta.bias)

    def forward(self, fused_bev: torch.Tensor, batch: dict, mode: str = "test") -> dict:
        base_predictions = self.detector_adapter(batch)
        context = self.context(fused_bev)
        box_deltas = self.box_delta(context).reshape(-1, self.num_classes, 7)
        score_deltas = self.score_delta(context)
        predictions = []
        for batch_index, prediction in enumerate(base_predictions):
            labels = prediction["labels_3d"].long().clamp(0, self.num_classes - 1)
            boxes = prediction["boxes_3d"]
            scores = prediction["scores_3d"].clamp(1e-5, 1.0 - 1e-5)
            if labels.numel():
                boxes = boxes + box_deltas[batch_index, labels]
                logits = torch.logit(scores) + score_deltas[batch_index, labels]
                scores = torch.sigmoid(logits)
            predictions.append(
                {"boxes_3d": boxes, "scores_3d": scores, "labels_3d": labels}
            )
        return {
            "predictions": predictions,
            "box_deltas": box_deltas,
            "score_deltas": score_deltas,
            "mode": mode,
        }
