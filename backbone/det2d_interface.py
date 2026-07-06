# -*- coding: utf-8 -*-
"""
Det2DInterface

2D 检测输出接口（仅定义格式，不实现真实检测）。
要求输出与 fusionrepo 的 iou2d_matrix / pair_selector / pair_feature_encoder 输入完全对齐。

核心约束：
- boxes_2d 必须是 (K,4) 且格式为 [x1, y1, x2, y2]（像素坐标）
- scores_2d 必须是 (K,)
- dtype/device 需与上层保持一致（通常 float32, same device）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


@dataclass
class Det2DOutput:
    """
    2D 检测输出的标准数据结构（必须满足下述形状/语义）。

    Attributes:
        boxes_2d: (K, 4) float
            2D bbox，像素坐标，格式严格为 [x1, y1, x2, y2]。
            - x1 < x2, y1 < y2（建议上游确保）
            - 与 fusionrepo::iou2d_matrix 直接对齐

        scores_2d: (K,) float
            每个 2D bbox 的置信度分数。

        labels_2d: (K,) long (可选)
            每个 2D bbox 的类别 id。若你的融合暂时不区分类别，可先不提供。

        extra: dict (可选)
            额外信息（例如原图尺寸、缩放比例、mmdet meta 等），不参与算子输入对齐，
            但可以方便后续 debug 或可视化。
    """
    boxes_2d: torch.Tensor
    scores_2d: torch.Tensor
    labels_2d: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None


class Det2DInterface(nn.Module):
    """
    2D 检测接口（抽象类）

    forward() 输入输出规范：
    - 输入 images: (B, 3, H, W) float tensor（或你自己的图像 batch 形式）
    - 输出 Det2DOutput，其中 boxes_2d/scores_2d 必须满足 CLOCs 融合算子的格式要求

    后续你写真实实现（比如基于 mmdetection）时：
    - 继承该类
    - 在 forward 中调用 mmdet
    - 再把 mmdet 的 bbox 输出转换为本接口规定的 Det2DOutput
    """

    def forward(self, images: torch.Tensor, image_meta: Optional[Dict[str, Any]] = None) -> Det2DOutput:
        raise NotImplementedError("Det2DInterface is abstract. Provide a concrete implementation.")


def ensure_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    确保 boxes 是 [x1,y1,x2,y2] 格式。
    如果你后续实现拿到的是 [x,y,w,h] 或其它格式，请在真实实现里转换后再返回。

    这里只做最基本的检查（不做猜测转换，避免悄悄出错影响精度）。
    """
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes_2d must be (K,4), got {tuple(boxes.shape)}")
    return boxes


def validate_det2d_output(out: Det2DOutput) -> Det2DOutput:
    """
    对 Det2DOutput 做强约束检查，防止后续融合阶段出现“不对齐”的隐蔽 bug。
    建议在你的 framework/module 里调试阶段启用。
    """
    boxes = ensure_xyxy(out.boxes_2d)
    scores = out.scores_2d
    if scores.dim() != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError(f"scores_2d must be (K,), K={boxes.shape[0]}, got {tuple(scores.shape)}")

    if out.labels_2d is not None:
        if out.labels_2d.dim() != 1 or out.labels_2d.shape[0] != boxes.shape[0]:
            raise ValueError("labels_2d must be (K,) and match boxes_2d")

    return out