# -*- coding: utf-8 -*-
"""
nms_bev

说明：
    将 BEV 平面上的 2D 框 NMS 注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.nms_bev(boxes_bev, scores, iou_threshold)

兼容性：
    - 兼容 PyTorch 2.0.1
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    对一组 BEV 平面二维框执行非极大值抑制（NMS），
    返回保留框的索引。

输入：
    boxes_bev: (N, 4)
        BEV 平面二维框，格式为 [x1, y1, x2, y2]
        注意：
            如果你的上游算子输出是 [x1, z1, x2, z2]，
            这里也可以直接传入，因为 NMS 只关心两个平面轴的矩形重叠关系。

    scores: (N,)
        每个框的置信度分数

    iou_threshold: float
        NMS 的 IoU 阈值

输出：
    keep_indices: (K,)
        保留下来的框索引（int64）
"""

from __future__ import annotations

import torch
from torch.library import Library




# ============================================================
# 2) IoU 计算（BEV 2D）
# ============================================================
def _bev_iou_single_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    计算单个 box 与多个 boxes 的 IoU

    Args:
        box:   (4,)   [x1, y1, x2, y2]
        boxes: (M,4)  [x1, y1, x2, y2]

    Returns:
        ious:  (M,)
    """
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    union = area_box + area_boxes - inter
    iou = inter / union.clamp(min=1e-12)
    return iou


# ============================================================
# 3) Python 实现
# ============================================================
def _nms_bev_impl(
    boxes_bev: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    具体实现：
        - boxes_bev: (N,4)
        - scores:    (N,)
    """
    if boxes_bev.dim() != 2 or boxes_bev.shape[1] != 4:
        raise ValueError("boxes_bev must have shape (N, 4)")
    if scores.dim() != 1:
        raise ValueError("scores must have shape (N,)")
    if boxes_bev.shape[0] != scores.shape[0]:
        raise ValueError("boxes_bev and scores must have the same first dimension")

    N = boxes_bev.shape[0]
    device = boxes_bev.device

    if N == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = _bev_iou_single_to_many(boxes_bev[i], boxes_bev[rest])
        order = rest[ious <= float(iou_threshold)]

    return torch.stack(keep).to(dtype=torch.int64)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("nms_bev", _nms_bev_impl)


# ============================================================
# 4) Meta 实现
# ============================================================
def _nms_bev_meta(
    boxes_bev: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    # NMS 输出长度不定，meta 下保守返回 (N,)
    return torch.empty((boxes_bev.shape[0],), device="meta", dtype=torch.int64)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("nms_bev", _nms_bev_meta)


# ============================================================
# 5) Python 包装
# ============================================================
def nms_bev(
    boxes_bev: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    return torch.ops.fusionrepo.nms_bev(
        boxes_bev,
        scores,
        float(iou_threshold)
    )