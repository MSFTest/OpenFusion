# -*- coding: utf-8 -*-
"""Build EPNet system with frozen MMDet/MMDet3D backbones."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.image_feature_interface import ImageFeatureInterface
from model.epnet.epnet_framework import EPNetFramework
from model.epnet.module.kitti_postprocess_module import KITTIPostProcessModule


MMDET_MODEL_NAME = "Faster R-CNN R50-Caffe-FPN"
MMDET_CONFIG = (
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/"
    "mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"
)
MMDET_CHECKPOINT = "checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth"
MMDET_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
    "faster_rcnn_r50_caffe_fpn_mstrain_1x_coco/"
    "faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth"
)

MMDET3D_MODEL_NAME = "PV-RCNN 8x2 80e KITTI 3-class"
MMDET3D_CONFIG = (
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/"
    "mmdet3d/.mim/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
)
MMDET3D_CHECKPOINT = "checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
MMDET3D_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
)
KITTI_CLASS_NAMES = ["Pedestrian", "Cyclist", "Car"]


def ensure_checkpoint(path: str, url: str, min_bytes: int) -> str:
    """检查 checkpoint，不存在时下载，残缺文件直接报错。"""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if os.path.exists(abs_path):
        size = os.path.getsize(abs_path)
        if size >= int(min_bytes):
            return abs_path
        raise RuntimeError(
            f"Checkpoint is incomplete: {abs_path} ({size} bytes), expected >= {min_bytes}. "
            f"Please remove it and download from {url}."
        )
    try:
        urllib.request.urlretrieve(url, abs_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download checkpoint from {url} to {abs_path}: {exc}") from exc
    size = os.path.getsize(abs_path)
    if size < int(min_bytes):
        raise RuntimeError(f"Downloaded checkpoint is incomplete: {abs_path} ({size} bytes), url={url}")
    return abs_path


def _freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


class MMDetImageExtractor(nn.Module):
    """真实 MMDetection 图像特征适配器。"""

    def __init__(self, config: str, checkpoint: str, device: str):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet")
        self.model = init_detector(config, checkpoint, device=device)
        _freeze(self.model)

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        self.model.eval()
        return self.model.extract_feat(images)


class MMDet3DDetectorAdapter(nn.Module):
    """真实 MMDetection3D 点云检测适配器。"""

    def __init__(self, config: str, checkpoint: str, device: str):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet3d.apis import init_model
        from mmdet3d.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet3d")
        self.model = init_model(config, checkpoint, device=device)
        _freeze(self.model)
        self.device_name = device

    @staticmethod
    def _empty(device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "boxes_3d": torch.zeros((0, 7), dtype=torch.float32, device=device),
            "scores_3d": torch.zeros((0,), dtype=torch.float32, device=device),
            "labels_3d": torch.zeros((0,), dtype=torch.long, device=device),
        }

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self.model.eval()
        device = next(self.model.parameters()).device
        pcd_path = batch.get("pcd_path")
        if not pcd_path:
            return self._empty(device)
        from mmdet3d.apis import inference_detector

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
        result = inference_detector(self.model, pcd_path)
        if isinstance(result, tuple):
            result = result[0]
        pred = getattr(result, "pred_instances_3d", None)
        if pred is None:
            return self._empty(device)
        boxes_obj = getattr(pred, "bboxes_3d", None)
        scores = getattr(pred, "scores_3d", None)
        labels = getattr(pred, "labels_3d", None)
        if boxes_obj is None or scores is None or labels is None:
            return self._empty(device)
        boxes = boxes_obj.tensor if hasattr(boxes_obj, "tensor") else boxes_obj
        out = {
            "boxes_3d": boxes.to(device=device, dtype=torch.float32),
            "scores_3d": scores.to(device=device, dtype=torch.float32),
            "labels_3d": labels.to(device=device, dtype=torch.long),
        }
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
        return out


class EPNetCriterion(nn.Module):
    """
    点级真实监督损失。

    前景点由 LiDAR 坐标点是否落入任一 GT 3D 框近似得到，训练目标作用在 EPNet 新增
    alignment / fusion / point head 参数上，不更新冻结 detector。
    """

    def __init__(
        self,
        box_loss_weight: float = 0.1,
        proposal_cls_weight: float = 1.0,
        proposal_box_weight: float = 1.0,
        proposal_loc_weight: float = 0.5,
        consistency_weight: float = 0.2,
        proposal_pos_iou_thr: float = 0.55,
        proposal_neg_iou_thr: float = 0.25,
        force_match_min_iou: float = 0.15,
    ):
        super().__init__()
        self.box_loss_weight = float(box_loss_weight)
        self.proposal_cls_weight = float(proposal_cls_weight)
        self.proposal_box_weight = float(proposal_box_weight)
        self.proposal_loc_weight = float(proposal_loc_weight)
        self.consistency_weight = float(consistency_weight)
        self.proposal_pos_iou_thr = float(proposal_pos_iou_thr)
        self.proposal_neg_iou_thr = float(proposal_neg_iou_thr)
        self.force_match_min_iou = float(force_match_min_iou)

    @staticmethod
    def _points_in_axis_aligned_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if boxes is None or boxes.numel() == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        xyz = points[:, :3]
        centers = boxes[:, :3]
        half = boxes[:, 3:6].abs() / 2.0
        lower = centers - half
        upper = centers + half
        inside = (xyz[:, None, :] >= lower[None, :, :]) & (xyz[:, None, :] <= upper[None, :, :])
        return inside.all(dim=-1).any(dim=1)

    @staticmethod
    def _bev_iou_axis_aligned(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0 or gt_boxes is None or gt_boxes.numel() == 0:
            return boxes.new_zeros((boxes.shape[0], 0))
        b_half = boxes[:, 3:5].abs() / 2.0
        g_half = gt_boxes[:, 3:5].abs() / 2.0
        b_min = boxes[:, :2] - b_half
        b_max = boxes[:, :2] + b_half
        g_min = gt_boxes[:, :2] - g_half
        g_max = gt_boxes[:, :2] + g_half
        lt = torch.maximum(b_min[:, None, :], g_min[None, :, :])
        rb = torch.minimum(b_max[:, None, :], g_max[None, :, :])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        area_b = (b_max[:, 0] - b_min[:, 0]).clamp(min=0) * (b_max[:, 1] - b_min[:, 1]).clamp(min=0)
        area_g = (g_max[:, 0] - g_min[:, 0]).clamp(min=0) * (g_max[:, 1] - g_min[:, 1]).clamp(min=0)
        return inter / (area_b[:, None] + area_g[None, :] - inter + 1e-6)

    @staticmethod
    def _proposal_box_loss(refined_boxes: torch.Tensor, matched_gt: torch.Tensor) -> torch.Tensor:
        center_loss = F.smooth_l1_loss(refined_boxes[:, :3], matched_gt[:, :3])
        size_loss = F.smooth_l1_loss(
            torch.log(refined_boxes[:, 3:6].abs().clamp(min=1e-3)),
            torch.log(matched_gt[:, 3:6].abs().clamp(min=1e-3)),
        )
        yaw_diff = torch.atan2(
            torch.sin(refined_boxes[:, 6] - matched_gt[:, 6]),
            torch.cos(refined_boxes[:, 6] - matched_gt[:, 6]),
        )
        yaw_loss = F.smooth_l1_loss(yaw_diff, torch.zeros_like(yaw_diff))
        return center_loss + size_loss + yaw_loss

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        points = batch["points"]
        if points.dim() == 3:
            points0 = points[0]
        else:
            points0 = points
        gt_boxes = batch.get("gt_boxes_3d")
        if gt_boxes is not None:
            gt_boxes = gt_boxes.to(device=points0.device, dtype=points0.dtype)
        target = self._points_in_axis_aligned_boxes(points0, gt_boxes).float().unsqueeze(0)
        logits = outputs["point_cls_logits"]
        cls_loss = F.binary_cross_entropy_with_logits(logits, target.expand_as(logits))
        loc_conf = outputs["point_loc_conf"]
        loc_loss = F.binary_cross_entropy(loc_conf, target.expand_as(loc_conf))
        pos = target.bool().expand_as(logits)
        if pos.any():
            box_loss = F.smooth_l1_loss(outputs["point_box_delta"][pos], torch.zeros_like(outputs["point_box_delta"][pos]))
        else:
            box_loss = outputs["point_box_delta"].sum() * 0.0
        proposal_cls_loss = logits.new_zeros([])
        proposal_box_loss = logits.new_zeros([])
        proposal_loc_loss = logits.new_zeros([])
        consistency_loss = logits.new_zeros([])
        proposal_num = logits.new_zeros([])
        proposal_pos_num = logits.new_zeros([])
        proposal_neg_num = logits.new_zeros([])
        proposal_ignore_num = logits.new_zeros([])
        proposal_mean_iou = logits.new_zeros([])
        proposal_max_iou = logits.new_zeros([])
        prediction = outputs.get("prediction", {})
        detector_boxes = outputs.get("detector_prediction", {}).get("boxes_3d")
        proposal_logits = outputs.get("proposal_score_logits", prediction.get("proposal_score_logits"))
        proposal_loc_conf = outputs.get("proposal_loc_conf", prediction.get("proposal_loc_conf"))
        refined_boxes = outputs.get("refined_boxes_3d", prediction.get("boxes_3d"))
        if (
            detector_boxes is not None
            and proposal_logits is not None
            and proposal_loc_conf is not None
            and refined_boxes is not None
            and detector_boxes.numel() > 0
        ):
            if gt_boxes is None or gt_boxes.numel() == 0:
                target_iou = proposal_logits.new_zeros(proposal_logits.shape)
                valid = torch.ones_like(target_iou, dtype=torch.bool)
                matched_gt = None
            else:
                ious = self._bev_iou_axis_aligned(detector_boxes, gt_boxes)
                target_iou, matched_idx = ious.max(dim=1)
                if ious.numel() > 0:
                    gt_best_iou, force_props = ious.max(dim=0)
                    force_mask = gt_best_iou >= self.force_match_min_iou
                    if force_mask.any():
                        force_props = force_props[force_mask]
                        force_gts = torch.arange(gt_boxes.shape[0], device=gt_boxes.device)[force_mask]
                        target_iou[force_props] = torch.maximum(
                            target_iou[force_props],
                            target_iou.new_full((force_props.numel(),), self.proposal_pos_iou_thr),
                        )
                        matched_idx[force_props] = force_gts
                valid = (target_iou < self.proposal_neg_iou_thr) | (target_iou >= self.proposal_pos_iou_thr)
                matched_gt = gt_boxes[matched_idx]
            proposal_num = detector_boxes.new_tensor(float(detector_boxes.shape[0]))
            proposal_pos_num = (target_iou >= self.proposal_pos_iou_thr).sum().to(dtype=detector_boxes.dtype)
            proposal_neg_num = (target_iou < self.proposal_neg_iou_thr).sum().to(dtype=detector_boxes.dtype)
            proposal_ignore_num = proposal_num - proposal_pos_num - proposal_neg_num
            proposal_mean_iou = target_iou.mean() if target_iou.numel() else detector_boxes.new_zeros([])
            proposal_max_iou = target_iou.max() if target_iou.numel() else detector_boxes.new_zeros([])
            if valid.any():
                proposal_target = (target_iou >= self.proposal_pos_iou_thr).to(dtype=proposal_logits.dtype)
                proposal_cls_loss = F.binary_cross_entropy_with_logits(
                    proposal_logits[valid], proposal_target[valid]
                )
            proposal_loc_loss = F.smooth_l1_loss(proposal_loc_conf, target_iou.detach().clamp(0.0, 1.0))
            consistency_loss = F.smooth_l1_loss(torch.sigmoid(proposal_logits), proposal_loc_conf.detach())
            pos = target_iou >= self.proposal_pos_iou_thr
            if matched_gt is not None and pos.any():
                proposal_box_loss = self._proposal_box_loss(refined_boxes[pos], matched_gt[pos])
        total = (
            cls_loss
            + loc_loss
            + self.box_loss_weight * box_loss
            + self.proposal_cls_weight * proposal_cls_loss
            + self.proposal_box_weight * proposal_box_loss
            + self.proposal_loc_weight * proposal_loc_loss
            + self.consistency_weight * consistency_loss
        )
        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_loc": loc_loss.detach(),
            "loss_box": box_loss.detach(),
            "loss_proposal_cls": proposal_cls_loss.detach(),
            "loss_proposal_box": proposal_box_loss.detach(),
            "loss_proposal_loc": proposal_loc_loss.detach(),
            "loss_consistency": consistency_loss.detach(),
            "proposal_num": proposal_num.detach(),
            "proposal_pos_num": proposal_pos_num.detach(),
            "proposal_neg_num": proposal_neg_num.detach(),
            "proposal_ignore_num": proposal_ignore_num.detach(),
            "proposal_pos_ratio": (proposal_pos_num / proposal_num.clamp(min=1.0)).detach(),
            "proposal_mean_iou": proposal_mean_iou.detach(),
            "proposal_max_iou": proposal_max_iou.detach(),
        }


class EPNetPostProcessor(nn.Module):
    """KITTI label_2 后处理封装。"""

    def __init__(self, class_names: List[str]):
        super().__init__()
        self.kitti = KITTIPostProcessModule(class_names=class_names)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> List[str]:
        return self.kitti(outputs["prediction"], batch["calib"])


@dataclass
class EPNetSystem:
    model: EPNetFramework
    criterion: EPNetCriterion
    postprocessor: EPNetPostProcessor
    optimizer: torch.optim.Optimizer
    image_adapter: MMDetImageExtractor
    point_cloud_adapter: MMDet3DDetectorAdapter
    device: torch.device
    class_names: List[str]
    freeze_report: Dict[str, int]


def _count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()}
        else:
            out[key] = value
    return out


def build_epnet_system(
    device: str = "cuda:0",
    lr: float = 1e-3,
    proposal_pos_iou_thr: float = 0.55,
    proposal_neg_iou_thr: float = 0.25,
    proposal_box_weight: float = 1.0,
    proposal_cls_weight: float = 1.0,
    proposal_loc_weight: float = 0.5,
    consistency_weight: float = 0.2,
    force_match_min_iou: float = 0.15,
) -> EPNetSystem:
    """构建 EPNet 系统并验证冻结与 optimizer 参数。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    image_ckpt = ensure_checkpoint(MMDET_CHECKPOINT, MMDET_CHECKPOINT_URL, 100_000_000)
    point_ckpt = ensure_checkpoint(MMDET3D_CHECKPOINT, MMDET3D_CHECKPOINT_URL, 100_000_000)
    image_extractor = MMDetImageExtractor(MMDET_CONFIG, image_ckpt, device)
    image_adapter = ImageFeatureInterface(image_extractor, feat_index=0)
    point_adapter = MMDet3DDetectorAdapter(MMDET3D_CONFIG, point_ckpt, device)
    _freeze(image_adapter)
    _freeze(point_adapter)
    model = EPNetFramework(
        image_backbone=image_adapter,
        point_cloud_detector=point_adapter,
        point_in_channels=3,
        image_in_channels=256,
        fusion_channels=128,
    ).to(torch_device)
    _freeze(model.image_backbone)
    _freeze(model.point_cloud_detector)
    criterion = EPNetCriterion(
        proposal_cls_weight=proposal_cls_weight,
        proposal_box_weight=proposal_box_weight,
        proposal_loc_weight=proposal_loc_weight,
        consistency_weight=consistency_weight,
        proposal_pos_iou_thr=proposal_pos_iou_thr,
        proposal_neg_iou_thr=proposal_neg_iou_thr,
        force_match_min_iou=force_match_min_iou,
    ).to(torch_device)
    postprocessor = EPNetPostProcessor(KITTI_CLASS_NAMES).to(torch_device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    freeze_report = {
        "image_trainable": _count_trainable(model.image_backbone),
        "point_cloud_trainable": _count_trainable(model.point_cloud_detector),
        "optimizer_param_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
    }
    if freeze_report["image_trainable"] != 0 or freeze_report["point_cloud_trainable"] != 0:
        raise RuntimeError(f"Frozen backbone verification failed: {freeze_report}")
    if freeze_report["optimizer_param_count"] <= 0:
        raise RuntimeError("Optimizer has no trainable EPNet parameters.")
    return EPNetSystem(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        optimizer=optimizer,
        image_adapter=image_extractor,
        point_cloud_adapter=point_adapter,
        device=torch_device,
        class_names=KITTI_CLASS_NAMES,
        freeze_report=freeze_report,
    )


def train_step(system: EPNetSystem, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """执行一个真实训练 step。"""
    system.model.image_backbone.eval()
    system.model.point_cloud_detector.eval()
    system.model.train()
    system.model.image_backbone.eval()
    system.model.point_cloud_detector.eval()
    batch = _move_batch(batch, system.device)
    system.optimizer.zero_grad(set_to_none=True)
    outputs = system.model(batch, mode="train")
    losses = system.criterion(outputs, batch)
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError(f"EPNet loss is not finite: {losses['loss']}")
    if not losses["loss"].requires_grad:
        report = {key: value.detach() if torch.is_tensor(value) else value for key, value in losses.items()}
        report["skipped_no_grad"] = torch.ones((), device=system.device)
        del outputs
        del losses
        if system.device.type == "cuda":
            torch.cuda.empty_cache()
        return report
    losses["loss"].backward()
    system.optimizer.step()
    report = {key: value.detach() if torch.is_tensor(value) else value for key, value in losses.items()}
    report["skipped_no_grad"] = torch.zeros((), device=system.device)
    del outputs
    del losses
    if system.device.type == "cuda":
        torch.cuda.empty_cache()
    return report


@torch.no_grad()
def test_step(system: EPNetSystem, batch: Dict[str, Any]) -> Dict[str, Any]:
    """执行一个真实推理 step 并返回 KITTI label_2 行。"""
    system.model.eval()
    batch = _move_batch(batch, system.device)
    outputs = system.model(batch, mode="test")
    lines = system.postprocessor(outputs, batch)
    raw_outputs = dict(outputs)
    raw_outputs["prediction"] = outputs["detector_prediction"]
    raw_lines = system.postprocessor(raw_outputs, batch)
    return {"outputs": outputs, "label_2_lines": lines, "raw_label_2_lines": raw_lines}
