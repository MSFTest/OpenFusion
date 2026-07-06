# -*- coding: utf-8 -*-
"""
anchor_assign_iou

输入：
    anchors_3d: (N, 7) [x, y, z, dx, dy, dz, yaw]
    gt_boxes_3d: (G, 7) [x, y, z, dx, dy, dz, yaw]

输出：
    labels: (N,)
    bbox_targets: (N, 7)

说明：
    当前 IoU 仍采用 axis-aligned 方式，只使用前 6 维做体积 IoU；
    但回归 target 已经扩展到 7 维，包含 yaw。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _axis_aligned_iou3d_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    boxes1: (N,7) [x,y,z,dx,dy,dz,yaw]
    boxes2: (M,7) [x,y,z,dx,dy,dz,yaw]
    return: (N,M)

    IoU 计算忽略 yaw，仅按前 6 维 axis-aligned box 计算。
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=boxes1.dtype,
            device=boxes1.device
        )

    b1_min = boxes1[:, :3] - boxes1[:, 3:6] * 0.5
    b1_max = boxes1[:, :3] + boxes1[:, 3:6] * 0.5

    b2_min = boxes2[:, :3] - boxes2[:, 3:6] * 0.5
    b2_max = boxes2[:, :3] + boxes2[:, 3:6] * 0.5

    inter_min = torch.maximum(b1_min[:, None, :], b2_min[None, :, :])
    inter_max = torch.minimum(b1_max[:, None, :], b2_max[None, :, :])
    inter_size = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]

    vol1 = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5])[:, None]
    vol2 = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5])[None, :]

    union = vol1 + vol2 - inter_vol
    iou = inter_vol / union.clamp(min=1e-12)
    return iou


def _encode_box_targets(
    src_boxes: torch.Tensor,
    tgt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    简化版 7 维 target：
        target = gt - anchor
    """
    return tgt_boxes - src_boxes


def _anchor_assign_iou_impl(
    anchors_3d: torch.Tensor,
    gt_boxes_3d: torch.Tensor,
    pos_iou_thr: float,
    neg_iou_thr: float
):
    if anchors_3d.dim() != 2 or anchors_3d.shape[1] != 7:
        raise ValueError("anchors_3d must have shape (N,7)")

    if gt_boxes_3d.dim() != 2 or gt_boxes_3d.shape[1] != 7:
        raise ValueError("gt_boxes_3d must have shape (G,7)")

    if pos_iou_thr < neg_iou_thr:
        raise ValueError("pos_iou_thr should be >= neg_iou_thr")

    N = anchors_3d.shape[0]
    device = anchors_3d.device

    labels = torch.full((N,), -1, dtype=torch.long, device=device)
    bbox_targets = torch.zeros_like(anchors_3d)

    if N == 0:
        return labels, bbox_targets

    if gt_boxes_3d.shape[0] == 0:
        labels[:] = 0
        return labels, bbox_targets

    iou_mat = _axis_aligned_iou3d_matrix(anchors_3d, gt_boxes_3d)

    max_iou_per_anchor, matched_gt_index = torch.max(iou_mat, dim=1)
    _, matched_anchor_index = torch.max(iou_mat, dim=0)

    pos_mask = max_iou_per_anchor >= pos_iou_thr
    neg_mask = max_iou_per_anchor < neg_iou_thr

    labels[pos_mask] = 1
    labels[neg_mask] = 0

    # 保证每个 GT 至少一个正样本
    labels[matched_anchor_index] = 1

    positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(1)

    if positive_indices.numel() > 0:
        matched_gt = gt_boxes_3d[matched_gt_index[positive_indices]]
        matched_anchors = anchors_3d[positive_indices]
        bbox_targets[positive_indices] = _encode_box_targets(
            matched_anchors,
            matched_gt
        )

    return labels, bbox_targets


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("anchor_assign_iou", _anchor_assign_iou_impl)


def _anchor_assign_iou_meta(
    anchors_3d: torch.Tensor,
    gt_boxes_3d: torch.Tensor,
    pos_iou_thr: float,
    neg_iou_thr: float
):
    N = anchors_3d.shape[0]

    labels = torch.empty((N,), dtype=torch.long, device="meta")
    bbox_targets = torch.empty((N, 7), dtype=anchors_3d.dtype, device="meta")

    return labels, bbox_targets


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("anchor_assign_iou", _anchor_assign_iou_meta)


def anchor_assign_iou(
    anchors_3d: torch.Tensor,
    gt_boxes_3d: torch.Tensor,
    pos_iou_thr: float,
    neg_iou_thr: float
):
    return torch.ops.fusionrepo.anchor_assign_iou(
        anchors_3d,
        gt_boxes_3d,
        float(pos_iou_thr),
        float(neg_iou_thr)
    )