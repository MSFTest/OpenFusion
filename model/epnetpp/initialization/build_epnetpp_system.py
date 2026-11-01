# -*- coding: utf-8 -*-
"""Build EPNet++ system with frozen MMDet/MMDet3D models."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.image_feature_interface import ImageFeatureInterface
from model.epnet.module.kitti_postprocess_module import KITTIPostProcessModule
from model.epnetpp.epnetpp_framework import EPNetPPFramework
from model.epnetpp.module.mc_loss_module import MCLossModule


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
LOSS_VERSION = "epnetpp_residual_score_refine_v3"


def ensure_checkpoint(path: str, url: str, min_bytes: int) -> str:
    """检查 checkpoint；不存在则下载，残缺文件直接报错。"""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    if os.path.exists(abs_path):
        size = os.path.getsize(abs_path)
        if size >= int(min_bytes):
            return abs_path
        raise RuntimeError(
            f"Checkpoint is incomplete: {abs_path} ({size} bytes), expected >= {min_bytes}. "
            f"Remove it and download from {url}."
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
        self.class_names = list(KITTI_CLASS_NAMES)

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
        return {
            "boxes_3d": boxes.to(device=device, dtype=torch.float32),
            "scores_3d": scores.to(device=device, dtype=torch.float32),
            "labels_3d": labels.to(device=device, dtype=torch.long),
        }


class EPNetPPCriterion(nn.Module):
    """EPNet++ 真实监督损失，训练目标作用于新增 fusion/head 参数。"""

    def __init__(
        self,
        box_loss_weight: float = 0.1,
        mc_loss_weight: float = 0.2,
        point_cls_loss_weight: float = 1.0,
        point_loc_loss_weight: float = 1.0,
        score_refine_loss_weight: float = 0.5,
        score_preserve_loss_weight: float = 0.05,
        positive_iou_thr: float = 0.35,
    ):
        super().__init__()
        self.box_loss_weight = float(box_loss_weight)
        self.mc_loss_weight = float(mc_loss_weight)
        self.point_cls_loss_weight = float(point_cls_loss_weight)
        self.point_loc_loss_weight = float(point_loc_loss_weight)
        self.score_refine_loss_weight = float(score_refine_loss_weight)
        self.score_preserve_loss_weight = float(score_preserve_loss_weight)
        self.positive_iou_thr = float(positive_iou_thr)
        self.mc_loss = MCLossModule()

    @staticmethod
    def _points_in_axis_aligned_boxes(points: torch.Tensor, boxes: torch.Tensor | None) -> torch.Tensor:
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
    def _axis_aligned_iou_3d(boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0 or gt_boxes.numel() == 0:
            return boxes.new_zeros((boxes.shape[0], gt_boxes.shape[0]))
        box_half = boxes[:, 3:6].abs() / 2.0
        gt_half = gt_boxes[:, 3:6].abs() / 2.0
        box_min = boxes[:, :3] - box_half
        box_max = boxes[:, :3] + box_half
        gt_min = gt_boxes[:, :3] - gt_half
        gt_max = gt_boxes[:, :3] + gt_half
        inter_min = torch.maximum(box_min[:, None, :], gt_min[None, :, :])
        inter_max = torch.minimum(box_max[:, None, :], gt_max[None, :, :])
        inter = (inter_max - inter_min).clamp_min(0.0).prod(dim=-1)
        box_vol = (box_max - box_min).clamp_min(0.0).prod(dim=-1)
        gt_vol = (gt_max - gt_min).clamp_min(0.0).prod(dim=-1)
        union = box_vol[:, None] + gt_vol[None, :] - inter
        return inter / union.clamp_min(1e-6)

    def _score_refine_loss(self, outputs: Dict[str, torch.Tensor], gt_boxes: torch.Tensor | None) -> torch.Tensor:
        conf = outputs.get("score_refine_conf")
        pred = outputs.get("prediction", {})
        boxes = pred.get("boxes_3d")
        if conf is None or boxes is None or conf.numel() == 0 or gt_boxes is None or gt_boxes.numel() == 0:
            device = conf.device if torch.is_tensor(conf) else boxes.device if torch.is_tensor(boxes) else None
            return torch.zeros((), device=device)
        boxes = boxes.to(device=conf.device, dtype=conf.dtype)
        gt_boxes = gt_boxes.to(device=conf.device, dtype=conf.dtype)
        max_iou = self._axis_aligned_iou_3d(boxes, gt_boxes).max(dim=1).values
        target = (max_iou >= self.positive_iou_thr).to(conf.dtype)
        if target.sum() > 0 and target.sum() < target.numel():
            pos_weight = ((target.numel() - target.sum()) / target.sum()).clamp(1.0, 10.0)
            loss = F.binary_cross_entropy(conf.clamp(1e-4, 1.0 - 1e-4), target, reduction="none")
            weights = torch.where(target > 0, pos_weight, torch.ones_like(target))
            return (loss * weights).sum() / weights.sum().clamp_min(1.0)
        return F.binary_cross_entropy(conf.clamp(1e-4, 1.0 - 1e-4), target)

    @staticmethod
    def _score_preserve_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw = outputs.get("raw_scores_3d")
        pred = outputs.get("prediction", {})
        refined = pred.get("scores_3d")
        if raw is None or refined is None or raw.numel() == 0:
            device = raw.device if torch.is_tensor(raw) else None
            return torch.zeros((), device=device)
        return F.smooth_l1_loss(refined, raw.detach())

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        points = batch["points"][0] if batch["points"].dim() == 3 else batch["points"]
        gt_boxes = batch.get("gt_boxes_3d")
        if gt_boxes is not None:
            gt_boxes = gt_boxes.to(device=points.device, dtype=points.dtype)
        target = self._points_in_axis_aligned_boxes(points, gt_boxes).float().unsqueeze(0)
        logits = outputs["point_cls_logits"]
        cls_loss = F.binary_cross_entropy_with_logits(logits, target.expand_as(logits))
        loc_loss = F.binary_cross_entropy(outputs["point_loc_conf"], target.expand_as(logits))
        pos = target.bool().expand_as(logits)
        if pos.any():
            box_loss = F.smooth_l1_loss(outputs["point_box_delta"][pos], torch.zeros_like(outputs["point_box_delta"][pos]))
        else:
            box_loss = outputs["point_box_delta"].sum() * 0.0
        mc = self.mc_loss(logits, outputs["image_conf_map"], outputs["feature_xy_norm"])
        score_refine_loss = self._score_refine_loss(outputs, gt_boxes)
        score_preserve_loss = self._score_preserve_loss(outputs)
        total = (
            self.point_cls_loss_weight * cls_loss
            + self.point_loc_loss_weight * loc_loss
            + self.box_loss_weight * box_loss
            + self.mc_loss_weight * mc["loss_mc"]
            + self.score_refine_loss_weight * score_refine_loss
            + self.score_preserve_loss_weight * score_preserve_loss
        )
        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_loc": loc_loss.detach(),
            "loss_box": box_loss.detach(),
            "loss_mc": mc["loss_mc"].detach(),
            "loss_score_refine": score_refine_loss.detach(),
            "loss_score_preserve": score_preserve_loss.detach(),
        }


class EPNetPPPostProcessor(nn.Module):
    """KITTI label_2 后处理封装，2D bbox 由 3D box corners 投影得到。"""

    def __init__(self, class_names: List[str], score_thr: float = 0.001):
        super().__init__()
        self.kitti = KITTIPostProcessModule(class_names=class_names, score_thr=score_thr)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> List[str]:
        lines = self.kitti(outputs["prediction"], batch["calib"])
        image = batch.get("images")
        image_h = int(image.shape[-2]) if torch.is_tensor(image) else None
        image_w = int(image.shape[-1]) if torch.is_tensor(image) else None
        cleaned: List[str] = []
        for line in lines:
            parts = line.split()
            if len(parts) != 16:
                continue
            z = float(parts[13])
            if z <= 1e-3:
                continue
            if image_h is not None and image_w is not None:
                x1, y1, x2, y2 = [float(v) for v in parts[4:8]]
                x1 = max(0.0, min(float(image_w - 1), x1))
                x2 = max(0.0, min(float(image_w - 1), x2))
                y1 = max(0.0, min(float(image_h - 1), y1))
                y2 = max(0.0, min(float(image_h - 1), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                parts[4:8] = [f"{x1:.4f}", f"{y1:.4f}", f"{x2:.4f}", f"{y2:.4f}"]
            cleaned.append(" ".join(parts))
        return cleaned


@dataclass
class EPNetPPSystem:
    model: EPNetPPFramework
    criterion: EPNetPPCriterion
    postprocessor: EPNetPPPostProcessor
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


def build_epnetpp_system(
    device: str = "cuda:0",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    box_loss_weight: float = 0.1,
    mc_loss_weight: float = 0.2,
    point_cls_loss_weight: float = 1.0,
    point_loc_loss_weight: float = 1.0,
    score_refine_loss_weight: float = 0.5,
    score_preserve_loss_weight: float = 0.05,
    positive_iou_thr: float = 0.35,
    train_refine_only: bool = False,
) -> EPNetPPSystem:
    """构建 EPNet++ 系统并验证冻结与 optimizer 参数。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)
    image_ckpt = ensure_checkpoint(MMDET_CHECKPOINT, MMDET_CHECKPOINT_URL, 100_000_000)
    point_ckpt = ensure_checkpoint(MMDET3D_CHECKPOINT, MMDET3D_CHECKPOINT_URL, 100_000_000)
    image_extractor = MMDetImageExtractor(MMDET_CONFIG, image_ckpt, device)
    image_interface = ImageFeatureInterface(image_extractor, feat_index=0)
    point_adapter = MMDet3DDetectorAdapter(MMDET3D_CONFIG, point_ckpt, device)
    _freeze(image_interface)
    _freeze(point_adapter)
    model = EPNetPPFramework(
        image_backbone=image_interface,
        point_cloud_detector=point_adapter,
        image_in_channels=256,
        point_in_channels=3,
        fusion_channels=128,
    ).to(torch_device)
    _freeze(model.image_backbone)
    _freeze(model.point_adapter.point_cloud_detector)
    if train_refine_only:
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("head."))
    criterion = EPNetPPCriterion(
        box_loss_weight=box_loss_weight,
        mc_loss_weight=mc_loss_weight,
        point_cls_loss_weight=point_cls_loss_weight,
        point_loc_loss_weight=point_loc_loss_weight,
        score_refine_loss_weight=score_refine_loss_weight,
        score_preserve_loss_weight=score_preserve_loss_weight,
        positive_iou_thr=positive_iou_thr,
    ).to(torch_device)
    postprocessor = EPNetPPPostProcessor(KITTI_CLASS_NAMES).to(torch_device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    freeze_report = {
        "image_trainable": _count_trainable(model.image_backbone),
        "point_cloud_trainable": _count_trainable(model.point_adapter.point_cloud_detector),
        "optimizer_param_count": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
    }
    if train_refine_only:
        freeze_report["refine_only_trainable"] = _count_trainable(model.head)
    if freeze_report["image_trainable"] != 0 or freeze_report["point_cloud_trainable"] != 0:
        raise RuntimeError(f"Frozen backbone verification failed: {freeze_report}")
    if freeze_report["optimizer_param_count"] <= 0:
        raise RuntimeError("Optimizer has no trainable EPNet++ parameters.")
    return EPNetPPSystem(
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


def train_step(system: EPNetPPSystem, batch: Dict[str, Any], grad_clip: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """执行一个真实训练 step。"""
    system.model.train()
    system.model.image_backbone.eval()
    system.model.point_adapter.point_cloud_detector.eval()
    batch = _move_batch(batch, system.device)
    system.optimizer.zero_grad(set_to_none=True)
    outputs = system.model(batch, mode="train")
    losses = system.criterion(outputs, batch)
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError(f"EPNet++ loss is not finite: {losses['loss']}")
    if not losses["loss"].requires_grad:
        losses["skipped_no_grad"] = torch.ones((), device=system.device)
        return losses
    losses["loss"].backward()
    if grad_clip is not None and float(grad_clip) > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in system.model.parameters() if p.requires_grad],
            max_norm=float(grad_clip),
        )
        losses["grad_norm"] = grad_norm.detach()
    system.optimizer.step()
    losses["skipped_no_grad"] = torch.zeros((), device=system.device)
    return losses


@torch.no_grad()
def test_step(system: EPNetPPSystem, batch: Dict[str, Any]) -> Dict[str, Any]:
    """执行一个真实推理 step 并返回 KITTI label_2 行。"""
    system.model.eval()
    batch = _move_batch(batch, system.device)
    outputs = system.model(batch, mode="test")
    lines = system.postprocessor(outputs, batch)
    return {"outputs": outputs, "label_2_lines": lines}
