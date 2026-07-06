# -*- coding: utf-8 -*-
"""
roi_assign_iou

输入：
    proposals_3d : (N,7)
    gt_boxes_3d  : (G,7)

输出：
    labels       : (N,)
    bbox_targets : (N,7)

说明：
    当前 IoU 仍采用 axis-aligned 方式，只使用前 6 维参与 IoU；
    但 target 已经扩展到 7 维，包含 yaw。
"""

from __future__ import annotations
import torch
from torch.library import Library


def _iou3d(boxes1: torch.Tensor, boxes2: torch.Tensor):
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

    inter = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter[..., 0] * inter[..., 1] * inter[..., 2]

    vol1 = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5])[:, None]
    vol2 = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5])[None, :]

    union = vol1 + vol2 - inter_vol

    return inter_vol / union.clamp(min=1e-6)


def _roi_assign_iou_impl(
    proposals_3d,
    gt_boxes_3d,
    pos_iou_thr,
    neg_iou_thr,
):
    if proposals_3d.dim() != 2 or proposals_3d.shape[1] != 7:
        raise ValueError("proposals_3d must have shape (N,7)")
    if gt_boxes_3d.dim() != 2 or gt_boxes_3d.shape[1] != 7:
        raise ValueError("gt_boxes_3d must have shape (G,7)")

    N = proposals_3d.shape[0]

    labels = torch.full(
        (N,),
        -1,
        dtype=torch.long,
        device=proposals_3d.device
    )

    bbox_targets = torch.zeros_like(proposals_3d)

    if N == 0:
        return labels, bbox_targets

    if gt_boxes_3d.shape[0] == 0:
        labels[:] = 0
        return labels, bbox_targets

    ious = _iou3d(proposals_3d, gt_boxes_3d)
    max_iou, gt_index = ious.max(dim=1)

    pos_mask = max_iou >= pos_iou_thr
    neg_mask = max_iou < neg_iou_thr

    labels[pos_mask] = 1
    labels[neg_mask] = 0

    matched_gt = gt_boxes_3d[gt_index]
    bbox_targets[pos_mask] = matched_gt[pos_mask] - proposals_3d[pos_mask]

    return labels, bbox_targets


_lib_impl = Library("fusionrepo", "IMPL")
_lib_impl.impl("roi_assign_iou", _roi_assign_iou_impl, "CPU")
_lib_impl.impl("roi_assign_iou", _roi_assign_iou_impl, "CUDA")


def _roi_assign_iou_meta(
    proposals_3d,
    gt_boxes_3d,
    pos_iou_thr,
    neg_iou_thr,
):
    N = proposals_3d.shape[0]
    labels = torch.empty((N,), dtype=torch.long, device="meta")
    bbox_targets = torch.empty((N, 7), dtype=proposals_3d.dtype, device="meta")
    return labels, bbox_targets


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("roi_assign_iou", _roi_assign_iou_meta)


def roi_assign_iou(
    proposals_3d,
    gt_boxes_3d,
    pos_iou_thr,
    neg_iou_thr,
):
    return torch.ops.fusionrepo.roi_assign_iou(
        proposals_3d,
        gt_boxes_3d,
        float(pos_iou_thr),
        float(neg_iou_thr),
    )