# -*- coding: utf-8 -*-
"""
build_logonet_system.py

作用：
    1) 用 mmdetection 构建真实图像特征提取器
    2) 用 mmdetection3d 构建真实点云感知模块（点云特征提取 + 3D 框提取）
    3) 实例化 LoGoNetFramework
    4) 提供真实 train_step / test_step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import init_default_scope
from mmdet.utils import register_all_modules as register_mmdet_modules
from mmdet3d.utils import register_all_modules as register_mmdet3d_modules
from mmdet.apis import init_detector as init_det2d
from mmdet3d.apis import init_model as init_det3d

from backbone.image_feature_interface import ImageFeatureInterface
from backbone.point_cloud_perception_interface import PointCloudPerceptionInterface
from model.logonet.logonet_framework import LoGoNetFramework


register_mmdet_modules()
register_mmdet3d_modules()


# ============================================================
# 1) MMDet 图像特征提取适配器
# ============================================================
class MMDetImageExtractor(nn.Module):
    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str = "cuda:0",
        feat_index: Optional[int] = None
    ):
        super().__init__()
        self.device = device
        self.feat_index = feat_index

        init_default_scope("mmdet")
        self.model = init_det2d(cfg_path, ckpt_path, device=device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        init_default_scope("mmdet")

        if not hasattr(self.model, "extract_feat"):
            raise AttributeError("mmdet model must provide extract_feat(images)")

        feats = self.model.extract_feat(images)

        if self.feat_index is None:
            return feats

        if isinstance(feats, (list, tuple)):
            return feats[self.feat_index]

        return feats


# ============================================================
# 2) Box helpers
# ============================================================
def _limit_period(val: torch.Tensor, offset: float = 0.5, period: float = 2 * math.pi) -> torch.Tensor:
    return val - torch.floor(val / period + offset) * period


def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _xyzhwlr_to_bev_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))

    x = boxes[:, 0]
    y = boxes[:, 1]
    dx = boxes[:, 3]
    dy = boxes[:, 4]

    x1 = x - dx / 2.0
    y1 = y - dy / 2.0
    x2 = x + dx / 2.0
    y2 = y + dy / 2.0

    return torch.stack([x1, y1, x2, y2], dim=1)


def _bev_box_corners_np(box) -> list:
    x, y, _, dx, dy, _, yaw = [float(v) for v in box[:7]]
    hx = dx * 0.5
    hy = dy * 0.5
    local = [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


def _polygon_area(poly: list) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _inside_edge(p, a, b) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) <= 1e-8


def _line_intersection(p1, p2, q1, q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def _polygon_clip(subject: list, clip: list) -> list:
    output = subject
    for i in range(len(clip)):
        cp1 = clip[i]
        cp2 = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if len(input_list) == 0:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside_edge(e, cp1, cp2):
                if not _inside_edge(s, cp1, cp2):
                    output.append(_line_intersection(s, e, cp1, cp2))
                output.append(e)
            elif _inside_edge(s, cp1, cp2):
                output.append(_line_intersection(s, e, cp1, cp2))
            s = e
    return output


def _box_iou_bev_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1 = boxes1.detach().cpu().float().numpy()
    b2 = boxes2.detach().cpu().float().numpy()
    polys1 = [_bev_box_corners_np(x) for x in b1]
    polys2 = [_bev_box_corners_np(x) for x in b2]
    areas1 = [_polygon_area(p) for p in polys1]
    areas2 = [_polygon_area(p) for p in polys2]

    out = boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    vals = []
    for i, p1 in enumerate(polys1):
        row = []
        for j, p2 in enumerate(polys2):
            inter_poly = _polygon_clip(p1, p2)
            inter = _polygon_area(inter_poly)
            union = areas1[i] + areas2[j] - inter
            row.append(0.0 if union <= 1e-8 else inter / union)
        vals.append(row)
    return torch.tensor(vals, device=boxes1.device, dtype=boxes1.dtype).view_as(out)


def _box_iou_2d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def _rotated_bev_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        cur = order[0]
        keep.append(cur)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = _box_iou_bev_rotated(boxes[cur:cur + 1], boxes[rest]).reshape(-1)
        order = rest[ious <= float(iou_thr)]

    return torch.stack(keep).to(device=boxes.device, dtype=torch.long)


def _encode_boxes(rois: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    eps = 1e-6

    px, py, pz, pdx, pdy, pdz, pyaw = rois.unbind(dim=1)
    gx, gy, gz, gdx, gdy, gdz, gyaw = gt_boxes.unbind(dim=1)

    dx = (gx - px) / (pdx.clamp(min=eps))
    dy = (gy - py) / (pdy.clamp(min=eps))
    dz = (gz - pz) / (pdz.clamp(min=eps))

    ddx = torch.log(gdx.clamp(min=eps) / pdx.clamp(min=eps))
    ddy = torch.log(gdy.clamp(min=eps) / pdy.clamp(min=eps))
    ddz = torch.log(gdz.clamp(min=eps) / pdz.clamp(min=eps))

    dyaw = _angle_diff(gyaw, pyaw)

    return torch.stack([dx, dy, dz, ddx, ddy, ddz, dyaw], dim=1)


def _decode_boxes(rois: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    px, py, pz, pdx, pdy, pdz, pyaw = rois.unbind(dim=1)
    dx, dy, dz, ddx, ddy, ddz, dyaw = deltas.unbind(dim=1)

    gx = px + dx * pdx
    gy = py + dy * pdy
    gz = pz + dz * pdz

    gdx = pdx * torch.exp(ddx)
    gdy = pdy * torch.exp(ddy)
    gdz = pdz * torch.exp(ddz)

    gyaw = _limit_period(pyaw + dyaw, offset=0.5, period=2 * math.pi)

    return torch.stack([gx, gy, gz, gdx, gdy, gdz, gyaw], dim=1)


def _rotate_points_along_z(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot = torch.stack(
        [
            torch.stack([cosa, -sina], dim=-1),
            torch.stack([sina, cosa], dim=-1),
        ],
        dim=-2
    )
    xy = points[..., 0:2]
    xy_rot = torch.matmul(xy.unsqueeze(-2), rot).squeeze(-2)
    out = points.clone()
    out[..., 0:2] = xy_rot
    return out


def _encode_boxes_canonical(rois: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if rois.numel() == 0:
        return rois.new_zeros((0, 7))

    gt_local = gt_boxes.clone()
    roi_center = rois[:, 0:3]
    roi_yaw = _limit_period(rois[:, 6], offset=0.5, period=2 * math.pi)

    gt_local[:, 0:3] = gt_local[:, 0:3] - roi_center
    gt_local[:, 6] = gt_local[:, 6] - roi_yaw
    gt_local[:, 0:3] = _rotate_points_along_z(gt_local[:, 0:3], -roi_yaw)

    heading = _limit_period(gt_local[:, 6], offset=0.5, period=2 * math.pi)
    opposite_flag = (heading > math.pi * 0.5) & (heading < math.pi * 1.5)
    heading = heading.clone()
    heading[opposite_flag] = _limit_period(heading[opposite_flag] + math.pi, offset=0.5, period=2 * math.pi)
    heading = torch.where(heading > math.pi, heading - 2 * math.pi, heading)
    heading = torch.clamp(heading, min=-math.pi / 2, max=math.pi / 2)
    gt_local[:, 6] = heading

    rois_anchor = rois.clone()
    rois_anchor[:, 0:3] = 0
    rois_anchor[:, 6] = 0
    return _encode_boxes(rois_anchor, gt_local)


def _decode_boxes_canonical(rois: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    if rois.numel() == 0:
        return rois.new_zeros((0, 7))

    rois_anchor = rois.clone()
    rois_anchor[:, 0:3] = 0
    rois_anchor[:, 6] = 0
    decoded_local = _decode_boxes(rois_anchor, deltas)
    decoded_global = decoded_local.clone()
    decoded_global[:, 0:3] = _rotate_points_along_z(decoded_local[:, 0:3], rois[:, 6])
    decoded_global[:, 0:3] = decoded_global[:, 0:3] + rois[:, 0:3]
    decoded_global[:, 6] = _limit_period(decoded_local[:, 6] + rois[:, 6], offset=0.5, period=2 * math.pi)
    return decoded_global


def _boxes_to_corners_3d(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8, 3))

    x, y, z, dx, dy, dz, yaw = boxes.unbind(dim=1)
    template = boxes.new_tensor(
        [
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )
    corners = template.unsqueeze(0) * boxes[:, None, 3:6]
    corners = _rotate_points_along_z(corners.view(-1, 3), yaw.repeat_interleave(8)).view(-1, 8, 3)
    corners = corners + boxes[:, None, 0:3]
    return corners


def _corner_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return pred_boxes.new_tensor(0.0)

    pred_corners = _boxes_to_corners_3d(pred_boxes)
    gt_corners = _boxes_to_corners_3d(gt_boxes)
    gt_flip = gt_boxes.clone()
    gt_flip[:, 6] = _limit_period(gt_flip[:, 6] + math.pi, offset=0.5, period=2 * math.pi)
    gt_flip_corners = _boxes_to_corners_3d(gt_flip)

    corner_dist = torch.min(
        torch.norm(pred_corners - gt_corners, dim=-1),
        torch.norm(pred_corners - gt_flip_corners, dim=-1)
    )
    return F.smooth_l1_loss(corner_dist, corner_dist.new_zeros(corner_dist.shape), reduction="mean")


def _sum_loss_dict(loss_dict: Dict[str, Any]) -> torch.Tensor:
    total = None
    for v in loss_dict.values():
        if isinstance(v, torch.Tensor):
            cur = v.mean()
        elif isinstance(v, list):
            cur = sum([x.mean() for x in v if isinstance(x, torch.Tensor)])
        else:
            continue

        total = cur if total is None else total + cur

    if total is None:
        total = torch.tensor(0.0, device="cpu")
    return total


# ============================================================
# 3) LoGoNet second-stage criterion
# ============================================================
class LoGoNetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        alpha: float = 1.0,
        pos_iou_thr: float = 0.55,
        neg_iou_thr: float = 0.25,
        roi_per_image: int = 128,
        fg_ratio: float = 0.5,
        cls_fg_thresh: float = 0.75,
        cls_bg_thresh: float = 0.25,
        cls_bg_thresh_lo: float = 0.1,
        reg_fg_thresh: float = 0.55,
        hard_bg_ratio: float = 0.8,
        cls_score_type: str = "roi_iou",
        force_match_per_gt: bool = True,
        code_weights: Optional[List[float]] = None,
        use_roi_score_fusion_in_loss: bool = False
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.pos_iou_thr = float(pos_iou_thr)
        self.neg_iou_thr = float(neg_iou_thr)
        self.roi_per_image = int(roi_per_image)
        self.fg_ratio = float(fg_ratio)
        self.cls_fg_thresh = float(cls_fg_thresh)
        self.cls_bg_thresh = float(cls_bg_thresh)
        self.cls_bg_thresh_lo = float(cls_bg_thresh_lo)
        self.reg_fg_thresh = float(reg_fg_thresh)
        self.hard_bg_ratio = float(hard_bg_ratio)
        self.cls_score_type = str(cls_score_type)
        self.force_match_per_gt = bool(force_match_per_gt)
        self.use_roi_score_fusion_in_loss = bool(use_roi_score_fusion_in_loss)

        if code_weights is None:
            code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32))

    def _sample_bg_inds(
        self,
        hard_bg_inds: torch.Tensor,
        easy_bg_inds: torch.Tensor,
        bg_rois_per_this_image: int,
        device: torch.device
    ) -> torch.Tensor:
        if bg_rois_per_this_image <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)

        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * self.hard_bg_ratio), int(hard_bg_inds.numel()))
            easy_bg_rois_num = max(bg_rois_per_this_image - hard_bg_rois_num, 0)

            hard_perm = torch.randint(0, int(hard_bg_inds.numel()), (hard_bg_rois_num,), device=device)
            easy_perm = torch.randint(0, int(easy_bg_inds.numel()), (easy_bg_rois_num,), device=device)
            return torch.cat([hard_bg_inds[hard_perm], easy_bg_inds[easy_perm]], dim=0)

        if hard_bg_inds.numel() > 0:
            hard_perm = torch.randint(0, int(hard_bg_inds.numel()), (bg_rois_per_this_image,), device=device)
            return hard_bg_inds[hard_perm]

        if easy_bg_inds.numel() > 0:
            easy_perm = torch.randint(0, int(easy_bg_inds.numel()), (bg_rois_per_this_image,), device=device)
            return easy_bg_inds[easy_perm]

        return torch.empty((0,), device=device, dtype=torch.long)

    def _subsample_rois(
        self,
        max_iou: torch.Tensor
    ) -> torch.Tensor:
        device = max_iou.device

        if max_iou.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)

        fg_rois_per_image = int(round(self.fg_ratio * self.roi_per_image))
        fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)

        fg_inds = ((max_iou >= fg_thresh)).nonzero(as_tuple=False).view(-1)
        easy_bg_inds = ((max_iou < self.cls_bg_thresh_lo)).nonzero(as_tuple=False).view(-1)
        hard_bg_inds = ((max_iou < self.reg_fg_thresh) & (max_iou >= self.cls_bg_thresh_lo)).nonzero(as_tuple=False).view(-1)

        fg_num_rois = int(fg_inds.numel())
        bg_num_rois = int(hard_bg_inds.numel() + easy_bg_inds.numel())

        if fg_num_rois > 0 and bg_num_rois > 0:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            fg_perm = torch.randperm(fg_num_rois, device=device)[:fg_rois_per_this_image]
            fg_inds = fg_inds[fg_perm]

            bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
            bg_inds = self._sample_bg_inds(
                hard_bg_inds=hard_bg_inds,
                easy_bg_inds=easy_bg_inds,
                bg_rois_per_this_image=bg_rois_per_this_image,
                device=device
            )
            return torch.cat((fg_inds, bg_inds), dim=0)

        if fg_num_rois > 0:
            fg_perm = torch.randint(0, fg_num_rois, (self.roi_per_image,), device=device)
            return fg_inds[fg_perm]

        if bg_num_rois > 0:
            return self._sample_bg_inds(
                hard_bg_inds=hard_bg_inds,
                easy_bg_inds=easy_bg_inds,
                bg_rois_per_this_image=self.roi_per_image,
                device=device
            )

        return torch.empty((0,), device=device, dtype=torch.long)

    def _assign_targets_single(
        self,
        rois: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = rois.device
        num_rois = rois.shape[0]

        if num_rois == 0 or gt_boxes.numel() == 0:
            cls_targets = torch.empty((0,), device=device, dtype=rois.dtype)
            reg_targets = torch.empty((0, 7), device=device, dtype=rois.dtype)
            pos_mask = torch.empty((0,), device=device, dtype=torch.bool)
            sampled_inds = torch.empty((0,), device=device, dtype=torch.long)
            sampled_rois = torch.empty((0, 7), device=device, dtype=rois.dtype)
            matched_gt_boxes = torch.empty((0, 7), device=device, dtype=rois.dtype)
            return cls_targets, reg_targets, pos_mask, sampled_inds, sampled_rois, matched_gt_boxes

        ious = _box_iou_bev_rotated(rois, gt_boxes)
        max_iou, matched_gt = ious.max(dim=1)

        if self.force_match_per_gt and ious.numel() > 0:
            best_roi_per_gt = ious.argmax(dim=0)
            matched_gt[best_roi_per_gt] = torch.arange(
                gt_boxes.shape[0], device=device, dtype=torch.long
            )
            gt_arange = torch.arange(gt_boxes.shape[0], device=device, dtype=torch.long)
            max_iou = max_iou.clone()
            max_iou[best_roi_per_gt] = ious[best_roi_per_gt, gt_arange]

        sampled_inds = self._subsample_rois(max_iou)
        if sampled_inds.numel() == 0:
            cls_targets = torch.empty((0,), device=device, dtype=rois.dtype)
            reg_targets = torch.empty((0, 7), device=device, dtype=rois.dtype)
            pos_mask = torch.empty((0,), device=device, dtype=torch.bool)
            sampled_rois = torch.empty((0, 7), device=device, dtype=rois.dtype)
            matched_gt_boxes = torch.empty((0, 7), device=device, dtype=rois.dtype)
            return cls_targets, reg_targets, pos_mask, sampled_inds, sampled_rois, matched_gt_boxes

        sampled_iou = max_iou[sampled_inds]
        sampled_matched_gt = matched_gt[sampled_inds]
        sampled_rois = rois[sampled_inds]
        matched_gt_boxes = gt_boxes[sampled_matched_gt]

        pos_mask = sampled_iou > self.reg_fg_thresh
        reg_targets = torch.zeros((sampled_inds.numel(), 7), device=device, dtype=rois.dtype)
        if pos_mask.any():
            reg_targets[pos_mask] = _encode_boxes_canonical(
                sampled_rois[pos_mask],
                matched_gt_boxes[pos_mask]
            )

        if self.num_classes == 1 and self.cls_score_type == "roi_iou":
            fg_mask = sampled_iou > self.cls_fg_thresh
            bg_mask = sampled_iou < self.cls_bg_thresh
            interval_mask = (~fg_mask) & (~bg_mask)
            cls_targets = fg_mask.float()
            cls_targets[interval_mask] = (
                (sampled_iou[interval_mask] - self.cls_bg_thresh) /
                max(self.cls_fg_thresh - self.cls_bg_thresh, 1e-6)
            )
        else:
            cls_targets = torch.zeros((sampled_inds.numel(),), device=device, dtype=rois.dtype)
            cls_targets[pos_mask] = 1.0

        return cls_targets, reg_targets, pos_mask, sampled_inds, sampled_rois, matched_gt_boxes

    def forward(
        self,
        pred_out: Dict[str, Any],
        batch: Dict[str, Any],
        base_rpn_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        cls_score = pred_out["cls_score"]
        reg_offset = pred_out["reg_offset"]
        flat_rois = pred_out["flat_rois"]
        flat_roi_batch_indices = pred_out["flat_roi_batch_indices"]
        roi_scores = pred_out.get("roi_scores", None)

        gt_boxes = batch["gt_boxes"]
        gt_labels = batch["gt_labels"]

        if torch.is_tensor(gt_boxes):
            gt_boxes = [gt_boxes]
        if torch.is_tensor(gt_labels):
            gt_labels = [gt_labels]

        device = flat_rois.device
        cls_targets_all = []
        reg_targets_all = []
        pos_mask_all = []
        sampled_global_inds_all = []
        sampled_rois_all = []
        matched_gt_boxes_all = []

        batch_size = len(gt_boxes)

        for b in range(batch_size):
            roi_mask = (flat_roi_batch_indices == b)
            rois_b = flat_rois[roi_mask]

            gt_boxes_b = gt_boxes[b].to(device)
            gt_labels_b = gt_labels[b].to(device)

            cls_targets_b, reg_targets_b, pos_mask_b, sampled_inds_b, sampled_rois_b, matched_gt_boxes_b = self._assign_targets_single(
                rois_b,
                gt_boxes_b,
                gt_labels_b
            )

            cls_targets_all.append(cls_targets_b)
            reg_targets_all.append(reg_targets_b)
            pos_mask_all.append(pos_mask_b)
            global_inds_b = torch.nonzero(roi_mask, as_tuple=False).view(-1)
            if sampled_inds_b.numel() > 0:
                sampled_global_inds_all.append(global_inds_b[sampled_inds_b])
                sampled_rois_all.append(sampled_rois_b)
                matched_gt_boxes_all.append(matched_gt_boxes_b)

        if len(cls_targets_all) > 0:
            cls_targets = torch.cat(cls_targets_all, dim=0)
            reg_targets = torch.cat(reg_targets_all, dim=0)
            pos_mask = torch.cat(pos_mask_all, dim=0)
            sampled_global_inds = (
                torch.cat(sampled_global_inds_all, dim=0)
                if len(sampled_global_inds_all) > 0
                else torch.empty((0,), device=device, dtype=torch.long)
            )
            sampled_rois = (
                torch.cat(sampled_rois_all, dim=0)
                if len(sampled_rois_all) > 0
                else torch.empty((0, 7), device=device, dtype=flat_rois.dtype)
            )
            matched_gt_boxes = (
                torch.cat(matched_gt_boxes_all, dim=0)
                if len(matched_gt_boxes_all) > 0
                else torch.empty((0, 7), device=device, dtype=flat_rois.dtype)
            )
        else:
            cls_targets = torch.empty((0,), device=device, dtype=flat_rois.dtype)
            reg_targets = torch.empty((0, 7), device=device, dtype=flat_rois.dtype)
            pos_mask = torch.empty((0,), device=device, dtype=torch.bool)
            sampled_global_inds = torch.empty((0,), device=device, dtype=torch.long)
            sampled_rois = torch.empty((0, 7), device=device, dtype=flat_rois.dtype)
            matched_gt_boxes = torch.empty((0, 7), device=device, dtype=flat_rois.dtype)

        valid_mask = cls_targets >= 0

        if self.num_classes == 1:
            cls_pred = cls_score.view(-1)[sampled_global_inds]
            reg_offset = reg_offset[sampled_global_inds]

            if self.use_roi_score_fusion_in_loss and roi_scores is not None:
                if isinstance(roi_scores, list):
                    if len(roi_scores) == 0:
                        raise ValueError("roi_scores is empty")
                    if not all(torch.is_tensor(x) for x in roi_scores):
                        raise TypeError(
                            f"roi_scores must be Tensor or list[Tensor], got {[type(x) for x in roi_scores]}"
                        )
                    roi_scores = torch.cat([x.reshape(-1) for x in roi_scores], dim=0)
                elif torch.is_tensor(roi_scores):
                    roi_scores = roi_scores.reshape(-1)
                else:
                    raise TypeError(f"roi_scores must be Tensor or list[Tensor], got {type(roi_scores)}")

                if roi_scores.numel() != cls_score.view(-1).numel():
                    raise ValueError(
                        f"roi_scores numel ({roi_scores.numel()}) != cls_pred numel ({cls_score.view(-1).numel()})"
                    )

                roi_scores = roi_scores[sampled_global_inds]
                roi_logits = torch.logit(roi_scores.clamp(1e-4, 1 - 1e-4))
                cls_pred = cls_pred + roi_logits

            cls_tgt = cls_targets.float()

            if valid_mask.any():
                num_pos = int((cls_tgt[valid_mask] >= 0.5).sum().item())
                num_neg = int((cls_tgt[valid_mask] < 0.5).sum().item())
                pos_weight = float(num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0

                loss_conf = F.binary_cross_entropy_with_logits(
                    cls_pred[valid_mask],
                    cls_tgt[valid_mask],
                    reduction="mean",
                    pos_weight=cls_pred.new_tensor(pos_weight)
                )
            else:
                loss_conf = cls_pred.sum() * 0.0
        else:
            cls_score = cls_score[sampled_global_inds]
            reg_offset = reg_offset[sampled_global_inds]
            if valid_mask.any():
                loss_conf = F.cross_entropy(
                    cls_score[valid_mask],
                    cls_targets[valid_mask],
                    reduction="mean"
                )
            else:
                loss_conf = cls_score.sum() * 0.0

        if pos_mask.any():
            diff = F.smooth_l1_loss(
                reg_offset[pos_mask],
                reg_targets[pos_mask],
                reduction="none",
                beta=1.0
            )
            diff = diff * self.code_weights.view(1, -1).to(diff.device)
            loss_reg = diff.mean()
            decoded_boxes = _decode_boxes_canonical(sampled_rois[pos_mask], reg_offset[pos_mask])
            loss_corner = _corner_loss(decoded_boxes, matched_gt_boxes[pos_mask])
            loss_reg = loss_reg + loss_corner
        else:
            loss_reg = reg_offset.sum() * 0.0

        if base_rpn_loss is None:
            base_rpn_loss = loss_conf.new_tensor(0.0)

        total_loss = base_rpn_loss + loss_conf + self.alpha * loss_reg

        return {
            "loss": total_loss,
            "loss_rpn": base_rpn_loss,
            "loss_conf": loss_conf,
            "loss_reg": loss_reg
        }


# ============================================================
# 4) Post processor
# ============================================================
class LoGoNetPostProcessor(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        score_thr: float = 0.0,
        nms_thr: float = 0.5,
        pre_nms_topk: int = 300,
        post_nms_topk: int = 100,
        fuse_roi_score: bool = True
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        self.pre_nms_topk = int(pre_nms_topk)
        self.post_nms_topk = int(post_nms_topk)
        self.fuse_roi_score = bool(fuse_roi_score)

    def forward(
        self,
        pred_out: Dict[str, Any]
    ) -> List[Dict[str, torch.Tensor]]:
        cls_score = pred_out["cls_score"]
        reg_offset = pred_out["reg_offset"]
        flat_rois = pred_out["flat_rois"]
        flat_roi_batch_indices = pred_out["flat_roi_batch_indices"]
        roi_scores = pred_out.get("roi_scores", None)

        # The criterion trains regression targets in the canonical RoI frame
        # (_encode_boxes_canonical), so inference must decode with the same
        # convention. Using _decode_boxes here interprets learned offsets in
        # the global frame and degrades AP as training progresses.
        decoded_boxes = _decode_boxes_canonical(flat_rois, reg_offset)

        if self.num_classes == 1:
            scores = torch.sigmoid(cls_score.view(-1))
            labels = torch.zeros_like(scores, dtype=torch.long)
        else:
            probs = F.softmax(cls_score, dim=1)
            scores, labels = probs.max(dim=1)

        # ------------------------------------------------
        # 关键修复：保持原始 ROI score 融合逻辑，但先把 roi_scores
        # 统一整理成与 flat_rois 对齐的一维 Tensor
        # ------------------------------------------------
        if self.fuse_roi_score and roi_scores is not None:
            if isinstance(roi_scores, list):
                if len(roi_scores) == 0:
                    raise ValueError("roi_scores is empty")
                if not all(torch.is_tensor(x) for x in roi_scores):
                    raise TypeError(
                        f"roi_scores must be Tensor or list[Tensor], got {[type(x) for x in roi_scores]}"
                    )
                roi_scores = torch.cat([x.reshape(-1) for x in roi_scores], dim=0)
            elif torch.is_tensor(roi_scores):
                roi_scores = roi_scores.reshape(-1)
            else:
                raise TypeError(f"roi_scores must be Tensor or list[Tensor], got {type(roi_scores)}")

            if roi_scores.numel() != scores.numel():
                raise ValueError(
                    f"roi_scores numel ({roi_scores.numel()}) != scores numel ({scores.numel()})"
                )

            roi_scores = roi_scores.to(device=scores.device, dtype=scores.dtype)

            # 保持原本的几何融合，不改精度逻辑
            scores = torch.sqrt(
                scores.clamp(min=1e-6) * roi_scores.clamp(min=1e-6)
            )

        batch_size = int(flat_roi_batch_indices.max().item()) + 1 if flat_roi_batch_indices.numel() > 0 else 0
        final_results = []

        for b in range(batch_size):
            mask = (flat_roi_batch_indices == b)
            boxes_b = decoded_boxes[mask]
            scores_b = scores[mask]
            labels_b = labels[mask]

            if boxes_b.numel() == 0:
                final_results.append({
                    "boxes_3d": boxes_b,
                    "scores_3d": scores_b,
                    "labels_3d": labels_b
                })
                continue

            score_mask = scores_b >= self.score_thr
            boxes_b = boxes_b[score_mask]
            scores_b = scores_b[score_mask]
            labels_b = labels_b[score_mask]

            if boxes_b.numel() == 0:
                final_results.append({
                    "boxes_3d": boxes_b,
                    "scores_3d": scores_b,
                    "labels_3d": labels_b
                })
                continue

            if self.pre_nms_topk > 0 and scores_b.numel() > self.pre_nms_topk:
                topk = torch.topk(scores_b, k=self.pre_nms_topk)
                keep_topk = topk.indices
                boxes_b = boxes_b[keep_topk]
                scores_b = scores_b[keep_topk]
                labels_b = labels_b[keep_topk]

            keep = _rotated_bev_nms(boxes_b, scores_b, self.nms_thr)

            if self.post_nms_topk > 0:
                keep = keep[:self.post_nms_topk]

            final_results.append({
                "boxes_3d": boxes_b[keep],
                "scores_3d": scores_b[keep],
                "labels_3d": labels_b[keep]
            })

        return final_results


# ============================================================
# 5) System object
# ============================================================
@dataclass
class LoGoNetSystem:
    model: LoGoNetFramework
    optimizer: torch.optim.Optimizer
    criterion: LoGoNetCriterion
    postprocessor: LoGoNetPostProcessor
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    device: str = "cuda:0"
    grad_clip_norm: Optional[float] = None


# ============================================================
# 6) Build system
# ============================================================
def build_logonet_system(
    image_cfg: str,
    image_ckpt: str,
    point_cloud_cfg: str,
    point_cloud_ckpt: str,
    device: str = "cuda:0",

    image_feat_index: Optional[int] = 0,
    point_cloud_feat_index: Optional[int] = None,
    prediction_branch: str = "pts_bbox",
    use_voxelize: bool = True,
    point_cloud_allowed_labels: Optional[List[int]] = None,

    lidar_channels: int = 128,
    image_channels: int = 256,
    pos_channels: int = 128,
    roi_grid_size: int = 6,
    roi_pool_mode: str = "mean",
    num_heads: int = 4,
    ffn_ratio: int = 4,
    num_classes: int = 1,
    box_dim: int = 7,

    alpha: float = 1.0,
    pos_iou_thr: float = 0.55,
    neg_iou_thr: float = 0.25,
    roi_per_image: int = 128,
    fg_ratio: float = 0.5,
    cls_fg_thresh: float = 0.75,
    cls_bg_thresh: float = 0.25,
    cls_bg_thresh_lo: float = 0.1,
    reg_fg_thresh: float = 0.55,
    hard_bg_ratio: float = 0.8,
    cls_score_type: str = "roi_iou",
    score_thr: float = 0.0,
    nms_thr: float = 0.5,
    pre_nms_topk: int = 300,
    post_nms_topk: int = 100,
    lr: float = 4e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: Optional[float] = 10.0,

    freeze_image_backbone: bool = True,
    freeze_point_cloud_backbone: bool = True,
    use_roi_score_fusion_in_loss: bool = False,
    fuse_roi_score_in_test: bool = True,
    epochs: int = 50,
    steps_per_epoch: int = 1500,
) -> LoGoNetSystem:
    image_extractor = MMDetImageExtractor(
        cfg_path=image_cfg,
        ckpt_path=image_ckpt,
        device=device,
        feat_index=image_feat_index
    )

    image_interface = ImageFeatureInterface(
        extractor=image_extractor,
        feat_index=None
    )

    init_default_scope("mmdet3d")
    point_cloud_detector = init_det3d(
        point_cloud_cfg,
        point_cloud_ckpt,
        device=device
    )

    point_cloud_interface = PointCloudPerceptionInterface(
        detector=point_cloud_detector,
        feat_index=point_cloud_feat_index,
        use_voxelize=use_voxelize,
        prediction_branch=prediction_branch,
        allowed_labels=point_cloud_allowed_labels
    )

    model = LoGoNetFramework(
        image_backbone=image_interface,
        point_cloud_backbone=point_cloud_interface,
        lidar_channels=lidar_channels,
        image_channels=image_channels,
        pos_channels=pos_channels,
        roi_grid_size=roi_grid_size,
        roi_pool_mode=roi_pool_mode,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        num_classes=num_classes,
        box_dim=box_dim
    ).to(device)

    if freeze_image_backbone:
        for p in model.image_backbone.parameters():
            p.requires_grad_(False)

    if freeze_point_cloud_backbone:
        for p in model.point_cloud_backbone.parameters():
            p.requires_grad_(False)

    criterion = LoGoNetCriterion(
        num_classes=num_classes,
        alpha=alpha,
        pos_iou_thr=pos_iou_thr,
        neg_iou_thr=neg_iou_thr,
        roi_per_image=roi_per_image,
        fg_ratio=fg_ratio,
        cls_fg_thresh=cls_fg_thresh,
        cls_bg_thresh=cls_bg_thresh,
        cls_bg_thresh_lo=cls_bg_thresh_lo,
        reg_fg_thresh=reg_fg_thresh,
        hard_bg_ratio=hard_bg_ratio,
        cls_score_type=cls_score_type,
        force_match_per_gt=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        use_roi_score_fusion_in_loss=use_roi_score_fusion_in_loss
    ).to(device)

    postprocessor = LoGoNetPostProcessor(
        num_classes=num_classes,
        score_thr=score_thr,
        nms_thr=nms_thr,
        pre_nms_topk=pre_nms_topk,
        post_nms_topk=post_nms_topk,
        fuse_roi_score=fuse_roi_score_in_test
    ).to(device)

    train_params = [p for p in model.parameters() if p.requires_grad]
    if len(train_params) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        train_params,
        lr=lr,
        weight_decay=weight_decay
    )

    # Step-based cosine schedule. It is robust to slight steps_per_epoch
    # mismatch and avoids OneCycleLR throwing when the last partial batch
    # creates more optimizer steps than estimated.
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(total_steps), 1),
        eta_min=lr * 0.05
    )

    return LoGoNetSystem(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        postprocessor=postprocessor,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=grad_clip_norm
    )


# ============================================================
# 7) RPN loss helper
# ============================================================
def _compute_base_rpn_loss(
    sys: LoGoNetSystem,
    batch: Dict[str, Any]
) -> torch.Tensor:
    if not any(p.requires_grad for p in sys.model.point_cloud_backbone.parameters()):
        return torch.tensor(0.0, device=sys.device)

    detector = sys.model.point_cloud_backbone.detector

    if "inputs" in batch and "data_samples" in batch and hasattr(detector, "loss"):
        try:
            loss_dict = detector.loss(batch["inputs"], batch["data_samples"])
            return _sum_loss_dict(loss_dict).to(sys.device)
        except Exception:
            pass

    if "points" in batch and hasattr(detector, "loss"):
        try:
            maybe_inputs = {"points": batch["points"]}
            if "images" in batch:
                maybe_inputs["img"] = batch["images"]
            if "img_metas" in batch:
                maybe_inputs["img_metas"] = batch["img_metas"]

            if "data_samples" in batch:
                loss_dict = detector.loss(maybe_inputs, batch["data_samples"])
                return _sum_loss_dict(loss_dict).to(sys.device)
        except Exception:
            pass

    if not getattr(_compute_base_rpn_loss, "_warned_zero_loss", False):
        print("[WARN] point cloud backbone is trainable, but detector RPN loss could not be computed; returning 0.")
        setattr(_compute_base_rpn_loss, "_warned_zero_loss", True)
    return torch.tensor(0.0, device=sys.device)


# ============================================================
# 8) real train step
# ============================================================
def train_step(
    sys: LoGoNetSystem,
    batch: Dict[str, Any]
) -> Dict[str, Any]:
    sys.model.train()
    sys.optimizer.zero_grad(set_to_none=True)

    pred_out = sys.model.train_forward(batch)
    base_rpn_loss = _compute_base_rpn_loss(sys, batch)

    loss_dict = sys.criterion(
        pred_out=pred_out,
        batch=batch,
        base_rpn_loss=base_rpn_loss
    )

    loss = loss_dict["loss"]
    loss.backward()

    if sys.grad_clip_norm is not None and sys.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(sys.model.parameters(), max_norm=sys.grad_clip_norm)

    sys.optimizer.step()
    if sys.scheduler is not None:
        sys.scheduler.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "loss_rpn": float(loss_dict["loss_rpn"].detach().cpu().item()),
        "loss_conf": float(loss_dict["loss_conf"].detach().cpu().item()),
        "loss_reg": float(loss_dict["loss_reg"].detach().cpu().item()),
        "pred": pred_out
    }


# ============================================================
# 9) real test step
# ============================================================
@torch.no_grad()
def test_step(
    sys: LoGoNetSystem,
    batch: Dict[str, Any]
) -> Dict[str, Any]:
    sys.model.eval()

    pred_out = sys.model.test_forward(batch)
    final_results = sys.postprocessor(pred_out)

    return {
        "pred": pred_out,
        "final_results": final_results
    }
