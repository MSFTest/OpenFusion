# -*- coding: utf-8 -*-
"""
SemanticScoreModule

提供 PointPainting 所需的像素级语义分数图。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticScoreModule(nn.Module):
    """
    语义分数提供模块。

    优先使用 batch 中已有的 semantic_score_map；若不存在且提供了
    image_backbone，则优先调用其 semantic_score 生成真实语义图；
    不支持该接口时再用 image_backbone 输出的特征生成可运行的兜底分数图。

    输入:
        batch: 包含 images / semantic_score_map 的字典。

    输出:
        semantic_score_map: (B, H, W, C)
    """

    def __init__(self, image_backbone: nn.Module | None = None, num_semantic_classes: int = 5):
        super().__init__()
        self.image_backbone = image_backbone
        self.num_semantic_classes = int(num_semantic_classes)

    def forward(self, batch: dict) -> torch.Tensor:
        score_map = batch.get("semantic_score_map", None)
        if score_map is not None:
            if score_map.dim() == 3:
                score_map = score_map.unsqueeze(0)
            return score_map

        images = batch.get("images", None)
        if images is None:
            points = batch["points"]
            if isinstance(points, (list, tuple)):
                device = points[0].device
                dtype = points[0].dtype
            else:
                device = points.device
                dtype = points.dtype
            return torch.ones((1, 1, 1, self.num_semantic_classes), device=device, dtype=dtype) / self.num_semantic_classes

        if self.image_backbone is None:
            b, _, h, w = images.shape
            return torch.ones(
                (b, h, w, self.num_semantic_classes),
                device=images.device,
                dtype=images.dtype,
            ) / self.num_semantic_classes

        if hasattr(self.image_backbone, "semantic_score"):
            try:
                return self.image_backbone.semantic_score(images, self.num_semantic_classes)
            except Exception:
                pass

        feats = self.image_backbone(images)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if feats.dim() != 4:
            raise ValueError("image_backbone must return (B, C, H, W) features")

        logits = feats[:, : self.num_semantic_classes]
        if logits.shape[1] < self.num_semantic_classes:
            pad = torch.zeros(
                (logits.shape[0], self.num_semantic_classes - logits.shape[1], logits.shape[2], logits.shape[3]),
                device=logits.device,
                dtype=logits.dtype,
            )
            logits = torch.cat([logits, pad], dim=1)
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return torch.softmax(logits, dim=1).permute(0, 2, 3, 1).contiguous()
