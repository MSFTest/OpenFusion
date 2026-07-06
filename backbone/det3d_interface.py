# -*- coding: utf-8 -*-
"""
Det3DInterface

3D 检测输出接口（仅定义格式，不实现真实检测）。
要求输出与 fusionrepo 的 points_to_image / pair_feature_encoder / iou3d_matrix / nms3d 输入完全对齐。

核心约束：
- boxes_3d 必须是 (N,7) 且格式为 [x, y, z, w, l, h, yaw]，并且在 LiDAR 坐标系下
- scores_3d 必须是 (N,)
- labels_3d 建议提供（KITTI 评测/多类融合需要）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


@dataclass
class Det3DOutput:
    """
    3D 检测输出的标准数据结构（必须满足下述形状/语义）。

    Attributes:
        boxes_3d: (N, 7) float
            3D bbox，格式严格为 [x, y, z, w, l, h, yaw]。
            坐标系要求：LiDAR 坐标系（因为你的投影/融合算子链默认从 LiDAR 出发）
            - (x,y,z) 为中心点
            - (w,l,h) 为尺寸
            - yaw 为朝向角

        scores_3d: (N,) float
            每个 3D bbox 的置信度分数（建议范围 [0,1]，来自 detector 的 score）。

        labels_3d: (N,) long
            类别 id（KITTI 评测/多类检测强烈建议提供）。
            如果你当前只做单类 Car，也可以全填 0，但字段建议保留。

        extra: dict (可选)
            额外信息（例如 box3d 的原对象、速度、属性等），不参与算子输入对齐。
    """
    boxes_3d: torch.Tensor
    scores_3d: torch.Tensor
    labels_3d: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None


class Det3DInterface(nn.Module):
    """
    3D 检测接口（抽象类）

    forward() 输入输出规范：
    - 输入 points: (B, M, C) 或你自己的点云 batch 格式
    - 输出 Det3DOutput，其中 boxes_3d/scores_3d/labels_3d 必须满足 CLOCs 融合算子的格式要求

    后续写真实实现（比如基于 mmdet3d 或 OpenPCDet）时：
    - 继承该类
    - 在 forward 中调用真实 3D detector
    - 将 detector 输出（可能是 LiDARInstance3DBoxes / BaseInstance3DBoxes 等对象）
      转换为本接口规定的 (N,7) tensor 后再返回
    """

    def forward(self, points: torch.Tensor, batch_meta: Optional[Dict[str, Any]] = None) -> Det3DOutput:
        raise NotImplementedError("Det3DInterface is abstract. Provide a concrete implementation.")


def ensure_box7_lidar(boxes_3d: torch.Tensor) -> torch.Tensor:
    """
    确保 boxes_3d 是 (N,7) 且字段顺序为 [x,y,z,w,l,h,yaw]。
    """
    if boxes_3d.dim() != 2 or boxes_3d.shape[1] != 7:
        raise ValueError(f"boxes_3d must be (N,7), got {tuple(boxes_3d.shape)}")
    return boxes_3d


def ensure_scores(scores_3d: torch.Tensor, n: int) -> torch.Tensor:
    """
    确保 scores_3d 是 (N,)。
    """
    if scores_3d.dim() != 1 or scores_3d.shape[0] != n:
        raise ValueError(f"scores_3d must be (N,), N={n}, got {tuple(scores_3d.shape)}")
    return scores_3d


def ensure_labels(labels_3d: Optional[torch.Tensor], n: int, device: torch.device) -> torch.Tensor:
    """
    labels_3d 强烈建议提供；若为 None，则默认生成全 0（单类情况下可用）。
    """
    if labels_3d is None:
        return torch.zeros((n,), dtype=torch.long, device=device)

    if labels_3d.dim() != 1 or labels_3d.shape[0] != n:
        raise ValueError(f"labels_3d must be (N,), N={n}, got {tuple(labels_3d.shape)}")

    if labels_3d.dtype != torch.long:
        labels_3d = labels_3d.to(dtype=torch.long)

    return labels_3d


def validate_det3d_output(out: Det3DOutput, require_labels: bool = True) -> Det3DOutput:
    """
    对 Det3DOutput 做强约束检查，防止后续融合阶段出现“不对齐”的隐蔽 bug。

    Args:
        out: Det3DOutput
        require_labels: bool
            - True：强制要求 labels_3d 不为空（推荐，KITTI/多类必须）
            - False：允许 labels_3d 为空（会在内部补全为全 0，适合单类快速跑通）

    Returns:
        Det3DOutput（若 labels_3d 为空且 require_labels=False，会补全 labels_3d）
    """
    boxes = ensure_box7_lidar(out.boxes_3d)
    n = boxes.shape[0]

    scores = ensure_scores(out.scores_3d, n)

    if require_labels:
        if out.labels_3d is None:
            raise ValueError("labels_3d is required for KITTI/multi-class evaluation. Provide labels_3d.")
        labels = ensure_labels(out.labels_3d, n, device=boxes.device)
    else:
        labels = ensure_labels(out.labels_3d, n, device=boxes.device)

    # 保证 device 一致（避免后续索引时报错）
    if scores.device != boxes.device:
        scores = scores.to(device=boxes.device)
    if labels.device != boxes.device:
        labels = labels.to(device=boxes.device)

    out.scores_3d = scores
    out.labels_3d = labels

    return out