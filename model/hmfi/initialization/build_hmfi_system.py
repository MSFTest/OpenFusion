# -*- coding: utf-8 -*-
"""
build_hmfi_system.py

作用：
    1) 用 mmdetection 构建真实图像特征提取器
    2) 用 mmdetection3d 构建真实点云感知模块
    3) 实例化 HMFIFramework
    4) 提供真实 train_step / test_step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms as nms2d

from mmengine.registry import init_default_scope
from mmdet.utils import register_all_modules as register_mmdet_modules
from mmdet3d.utils import register_all_modules as register_mmdet3d_modules
from mmdet.apis import init_detector as init_det2d
from mmdet3d.apis import init_model as init_det3d

from backbone.image_feature_interface import ImageFeatureInterface
from backbone.point_cloud_feature_interface import PointCloudFeatureInterface
from model.hmfi.hmfi_framework import HMFIFramework


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
# 3) HMFI criterion (基于LoGoNetCriterion修改)
# ============================================================
class HMFICriterion(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        alpha: float = 1.2,
        pos_iou_thr: float = 0.45,
        neg_iou_thr: float = 0.20,
        force_match_per_gt: bool = True,
        code_weights: Optional[List[float]] = None,
        use_roi_score_fusion_in_loss: bool = False,
        vfim_loss_weight: float = 0.02,
        max_pos_weight: float = 10.0
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.pos_iou_thr = float(pos_iou_thr)
        self.neg_iou_thr = float(neg_iou_thr)
        self.force_match_per_gt = bool(force_match_per_gt)
        self.use_roi_score_fusion_in_loss = bool(use_roi_score_fusion_in_loss)
        self.vfim_loss_weight = float(vfim_loss_weight)
        self.max_pos_weight = float(max_pos_weight)

        if code_weights is None:
            code_weights = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0]
        self.register_buffer("code_weights", torch.tensor(code_weights, dtype=torch.float32))

    def _assign_targets_single(
        self,
        rois: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = rois.device
        num_rois = rois.shape[0]

        cls_targets = torch.zeros((num_rois,), device=device, dtype=torch.long)
        iou_targets = torch.zeros((num_rois,), device=device, dtype=rois.dtype)
        reg_targets = torch.zeros((num_rois, 7), device=device, dtype=rois.dtype)
        pos_mask = torch.zeros((num_rois,), device=device, dtype=torch.bool)

        if num_rois == 0 or gt_boxes.numel() == 0:
            return cls_targets, iou_targets, reg_targets, pos_mask

        roi_bev = _xyzhwlr_to_bev_xyxy(rois)
        gt_bev = _xyzhwlr_to_bev_xyxy(gt_boxes)

        ious = _box_iou_2d(roi_bev, gt_bev)
        max_iou, matched_gt = ious.max(dim=1)

        pos_mask = max_iou >= self.pos_iou_thr
        ign_mask = (max_iou < self.pos_iou_thr) & (max_iou > self.neg_iou_thr)

        if self.force_match_per_gt and ious.numel() > 0:
            best_roi_per_gt = ious.argmax(dim=0)
            pos_mask[best_roi_per_gt] = True
            ign_mask[best_roi_per_gt] = False
            matched_gt[best_roi_per_gt] = torch.arange(
                gt_boxes.shape[0], device=device, dtype=torch.long
            )

        if self.num_classes == 1:
            cls_targets[pos_mask] = 1
        else:
            cls_targets[pos_mask] = gt_labels[matched_gt[pos_mask]].long()
        iou_targets[pos_mask] = max_iou[pos_mask].clamp(0.0, 1.0)

        if pos_mask.any():
            reg_targets[pos_mask] = _encode_boxes(
                rois[pos_mask],
                gt_boxes[matched_gt[pos_mask]]
            )

        cls_targets[ign_mask] = -1
        return cls_targets, iou_targets, reg_targets, pos_mask

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

        gt_boxes = batch["gt_boxes"]
        gt_labels = batch["gt_labels"]

        if torch.is_tensor(gt_boxes):
            gt_boxes = [gt_boxes]
        if torch.is_tensor(gt_labels):
            gt_labels = [gt_labels]

        device = flat_rois.device
        cls_targets_all = []
        iou_targets_all = []
        reg_targets_all = []
        pos_mask_all = []

        batch_size = len(gt_boxes)

        for b in range(batch_size):
            roi_mask = (flat_roi_batch_indices == b)
            rois_b = flat_rois[roi_mask]

            gt_boxes_b = gt_boxes[b].to(device)
            gt_labels_b = gt_labels[b].to(device)

            cls_targets_b, iou_targets_b, reg_targets_b, pos_mask_b = self._assign_targets_single(
                rois_b,
                gt_boxes_b,
                gt_labels_b
            )

            cls_targets_all.append(cls_targets_b)
            iou_targets_all.append(iou_targets_b)
            reg_targets_all.append(reg_targets_b)
            pos_mask_all.append(pos_mask_b)

        if len(cls_targets_all) > 0:
            cls_targets = torch.cat(cls_targets_all, dim=0)
            iou_targets = torch.cat(iou_targets_all, dim=0)
            reg_targets = torch.cat(reg_targets_all, dim=0)
            pos_mask = torch.cat(pos_mask_all, dim=0)
        else:
            cls_targets = torch.empty((0,), device=device, dtype=torch.long)
            iou_targets = torch.empty((0,), device=device, dtype=flat_rois.dtype)
            reg_targets = torch.empty((0, 7), device=device, dtype=flat_rois.dtype)
            pos_mask = torch.empty((0,), device=device, dtype=torch.bool)

        valid_mask = cls_targets >= 0
        num_valid = int(valid_mask.sum().item())
        num_pos = int(pos_mask.sum().item())
        num_neg = int(((cls_targets == 0) & valid_mask).sum().item())

        if self.num_classes == 1:
            cls_pred = cls_score.view(-1)
            cls_tgt = torch.where(cls_targets == 1, iou_targets, cls_targets.float())

            if valid_mask.any():
                pos_weight = float(num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0
                pos_weight = min(pos_weight, self.max_pos_weight)

                loss_conf = F.binary_cross_entropy_with_logits(
                    cls_pred[valid_mask],
                    cls_tgt[valid_mask],
                    reduction="mean",
                    pos_weight=cls_pred.new_tensor(pos_weight)
                )
            else:
                loss_conf = cls_pred.sum() * 0.0
        else:
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
        else:
            loss_reg = reg_offset.sum() * 0.0

        if base_rpn_loss is None:
            base_rpn_loss = loss_conf.new_tensor(0.0)

        loss_vfim = pred_out.get("vfim_loss", None)
        if loss_vfim is None:
            loss_vfim = loss_conf.new_tensor(0.0)

        total_loss = base_rpn_loss + loss_conf + self.alpha * loss_reg + self.vfim_loss_weight * loss_vfim

        return {
            "loss": total_loss,
            "loss_rpn": base_rpn_loss,
            "loss_conf": loss_conf,
            "loss_reg": loss_reg,
            "loss_vfim": loss_vfim,
            "loss_feat": loss_vfim,
            "num_rois": loss_conf.new_tensor(float(flat_rois.shape[0])),
            "num_valid": loss_conf.new_tensor(float(num_valid)),
            "num_pos": loss_conf.new_tensor(float(num_pos)),
            "num_neg": loss_conf.new_tensor(float(num_neg)),
        }


# ============================================================
# 4) Post processor
# ============================================================
class HMFIPostProcessor(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        score_thr: float = 0.05,
        nms_thr: float = 0.55,
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
        roi_scores = pred_out.get("flat_roi_scores", None)

        decoded_boxes = _decode_boxes(flat_rois, reg_offset)

        if self.num_classes == 1:
            scores = torch.sigmoid(cls_score.view(-1))
            labels = torch.zeros_like(scores, dtype=torch.long)
        else:
            probs = F.softmax(cls_score, dim=1)
            scores, labels = probs.max(dim=1)

        if self.fuse_roi_score and roi_scores is not None and roi_scores.numel() == scores.numel():
            scores = torch.sqrt(scores.clamp_min(0.0) * roi_scores.to(scores.device).clamp_min(0.0))

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

            bev_boxes = _xyzhwlr_to_bev_xyxy(boxes_b)
            keep = nms2d(bev_boxes, scores_b, self.nms_thr)

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
class HMFISystem:
    model: HMFIFramework
    optimizer: torch.optim.Optimizer
    criterion: HMFICriterion
    postprocessor: HMFIPostProcessor
    device: str = "cuda:0"
    grad_clip_norm: Optional[float] = None
    scheduler: Optional[Any] = None
    debug_base_loss: bool = False
    freeze_point_cloud_backbone: bool = True


# ============================================================
# 6) Build system
# ============================================================
def build_hmfi_system(
    image_cfg: str,
    image_ckpt: str,
    point_cloud_cfg: str,
    point_cloud_ckpt: str,
    device: str = "cuda:0",

    image_feat_index: Optional[int] = 0,
    point_cloud_feat_index: Optional[int] = None,

    lidar_channels: int = 4,
    image_channels: int = 256,
    fusion_out_channels: int = 64,
    d_k: int = 64,
    d_v: int = 64,
    num_heads: int = 4,
    depth_mode: int = 1,
    downsample_factor: int = 4,
    depth_range: tuple = (-0.5, 70.0),
    num_depth_bins: int = 128,
    voxel_size: tuple = (0.05, 0.05, 0.1),
    point_cloud_range: tuple = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
    use_vfim: bool = False,
    vfim_loss_weight: float = 0.02,
    num_classes: int = 1,
    box_dim: int = 7,

    alpha: float = 1.2,
    pos_iou_thr: float = 0.45,
    neg_iou_thr: float = 0.20,
    code_weights: Optional[List[float]] = None,
    max_pos_weight: float = 10.0,
    score_thr: float = 0.05,
    nms_thr: float = 0.1,
    pre_nms_topk: int = 4096,
    post_nms_topk: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    grad_clip_norm: Optional[float] = 10.0,
    debug_base_loss: bool = False,

    freeze_image_backbone: bool = True,
    freeze_point_cloud_backbone: bool = True
) -> HMFISystem:
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

    point_cloud_interface = PointCloudFeatureInterface(
        extractor=point_cloud_detector,
        feat_index=point_cloud_feat_index,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range
    )

    model = HMFIFramework(
        image_backbone=image_interface,
        point_cloud_backbone=point_cloud_interface,
        lidar_channels=lidar_channels,
        image_channels=image_channels,
        fusion_out_channels=fusion_out_channels,
        d_k=d_k,
        d_v=d_v,
        num_heads=num_heads,
        depth_mode=depth_mode,
        downsample_factor=downsample_factor,
        depth_range=depth_range,
        num_depth_bins=num_depth_bins,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        use_vfim=use_vfim,
        num_classes=num_classes,
        box_dim=box_dim
    ).to(device)

    if freeze_image_backbone:
        for p in model.image_backbone.parameters():
            p.requires_grad_(False)

    if freeze_point_cloud_backbone:
        for p in model.point_cloud_backbone.parameters():
            p.requires_grad_(False)

    criterion = HMFICriterion(
        num_classes=num_classes,
        alpha=alpha,
        pos_iou_thr=pos_iou_thr,
        neg_iou_thr=neg_iou_thr,
        force_match_per_gt=True,
        code_weights=code_weights,
        vfim_loss_weight=vfim_loss_weight,
        max_pos_weight=max_pos_weight
    ).to(device)

    postprocessor = HMFIPostProcessor(
        num_classes=num_classes,
        score_thr=score_thr,
        nms_thr=nms_thr,
        pre_nms_topk=pre_nms_topk,
        post_nms_topk=post_nms_topk
    ).to(device)

    train_params = [p for p in model.parameters() if p.requires_grad]
    if len(train_params) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        train_params,
        lr=lr,
        weight_decay=weight_decay
    )

    return HMFISystem(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        postprocessor=postprocessor,
        device=device,
        grad_clip_norm=grad_clip_norm,
        debug_base_loss=debug_base_loss,
        freeze_point_cloud_backbone=freeze_point_cloud_backbone
    )


# ============================================================
# 7) RPN loss helper
# ============================================================
def _compute_base_rpn_loss(
    sys: HMFISystem,
    batch: Dict[str, Any]
) -> torch.Tensor:
    if sys.freeze_point_cloud_backbone:
        return torch.zeros(1, device=sys.device, requires_grad=False)

    detector = sys.model.point_cloud_backbone.extractor

    if "inputs" in batch and "data_samples" in batch and hasattr(detector, "loss"):
        try:
            loss_dict = detector.loss(batch["inputs"], batch["data_samples"])
            return _sum_loss_dict(loss_dict).to(sys.device)
        except Exception as e:
            if sys.debug_base_loss:
                print(f"[WARN] base detector loss(inputs/data_samples) failed: {type(e).__name__}: {e}")

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
        except Exception as e:
            if sys.debug_base_loss:
                print(f"[WARN] base detector loss(points/data_samples) failed: {type(e).__name__}: {e}")

    return torch.zeros(1, device=sys.device, requires_grad=True)


# ============================================================
# 8) real train step
# ============================================================
def train_step(
    sys: HMFISystem,
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

    roi_valid_counts = pred_out.get("roi_valid_counts", None)
    if torch.is_tensor(roi_valid_counts) and roi_valid_counts.numel() > 0:
        mean_roi_valid_counts = float(roi_valid_counts.float().mean().detach().cpu().item())
    else:
        mean_roi_valid_counts = 0.0

    return {
        "loss": float(loss.detach().cpu().item()),
        "loss_rpn": float(loss_dict["loss_rpn"].detach().cpu().item()),
        "loss_conf": float(loss_dict["loss_conf"].detach().cpu().item()),
        "loss_reg": float(loss_dict["loss_reg"].detach().cpu().item()),
        "loss_vfim": float(loss_dict["loss_vfim"].detach().cpu().item()),
        "loss_feat": float(loss_dict["loss_feat"].detach().cpu().item()),
        "num_rois": float(loss_dict["num_rois"].detach().cpu().item()),
        "num_valid": float(loss_dict["num_valid"].detach().cpu().item()),
        "num_pos": float(loss_dict["num_pos"].detach().cpu().item()),
        "num_neg": float(loss_dict["num_neg"].detach().cpu().item()),
        "mean_roi_valid_counts": mean_roi_valid_counts,
        "pred": pred_out
    }


# ============================================================
# 9) real test step
# ============================================================
@torch.no_grad()
def test_step(
    sys: HMFISystem,
    batch: Dict[str, Any]
) -> Dict[str, Any]:
    sys.model.eval()

    pred_out = sys.model.test_forward(batch)
    final_results = sys.postprocessor(pred_out)

    return {
        "pred": pred_out,
        "final_results": final_results
    }
