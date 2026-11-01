# -*- coding: utf-8 -*-
"""Faraway-Frustum 真实系统构建、训练、测试与 KITTI 后处理。"""

from __future__ import annotations

import math
import os
import subprocess
import urllib.request
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.faraway_frustum.faraway_frustum_framework import FarawayFrustumFramework


def _package_config_path(package_name: str, relative_path: str, fallback_prefix: str) -> str:
    spec = importlib.util.find_spec(package_name)
    if spec is not None and spec.origin:
        package_root = os.path.dirname(spec.origin)
        candidate = os.path.join(package_root, ".mim", "configs", relative_path)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(fallback_prefix, package_name, ".mim", "configs", relative_path)


MMDET_MODEL_NAME = "mask-rcnn_r50_fpn_2x_coco"
MMDET_CONFIG = _package_config_path(
    "mmdet",
    "mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py",
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages",
)
MMDET_CHECKPOINT = (
    "checkpoints/"
    "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
)
MMDET_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/"
    "mask_rcnn_r50_fpn_2x_coco/"
    "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"
)
MMDET_MIN_BYTES = 120 * 1024 * 1024

MMDET3D_MODEL_NAME = "pv_rcnn_8xb2-80e_kitti-3d-3class"
MMDET3D_CONFIG = _package_config_path(
    "mmdet3d",
    "pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py",
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages",
)
MMDET3D_CHECKPOINT = "checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
MMDET3D_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
)
MMDET3D_MIN_BYTES = 100 * 1024 * 1024
MMDET3D_PARTA2_CAR_MODEL_NAME = "parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car"
MMDET3D_PARTA2_CAR_CONFIG = _package_config_path(
    "mmdet3d",
    "parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car.py",
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages",
)
MMDET3D_PARTA2_CAR_CHECKPOINT = "checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth"
MMDET3D_PARTA2_CAR_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/"
    "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car/"
    "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth"
)
MMDET3D_PARTA2_CAR_MIN_BYTES = 150 * 1024 * 1024
POINT_CLOUD_MODEL_SPECS = {
    "pv_rcnn_3class": (
        MMDET3D_MODEL_NAME,
        MMDET3D_CONFIG,
        MMDET3D_CHECKPOINT,
        MMDET3D_CHECKPOINT_URL,
        MMDET3D_MIN_BYTES,
        None,
    ),
    "parta2_car": (
        MMDET3D_PARTA2_CAR_MODEL_NAME,
        MMDET3D_PARTA2_CAR_CONFIG,
        MMDET3D_PARTA2_CAR_CHECKPOINT,
        MMDET3D_PARTA2_CAR_CHECKPOINT_URL,
        MMDET3D_PARTA2_CAR_MIN_BYTES,
        (2,),
    ),
}
KITTI_CLASS_NAMES = ("Pedestrian", "Cyclist", "Car")
LOSS_VERSION = "faraway_frustum_trainable_near_score_ranking_v13"


@dataclass(frozen=True)
class PointCloudModelSpec:
    """点云检测器实例化信息。"""

    name: str
    config: str
    checkpoint: str
    checkpoint_url: str
    min_bytes: int
    label_map: Optional[Tuple[int, ...]]


def resolve_point_cloud_spec(
    model_name: str,
    config: str = "",
    checkpoint: str = "",
    min_bytes: int = 10 * 1024 * 1024,
) -> PointCloudModelSpec:
    """解析默认或 Faraway 专用点云检测器权重，默认行为保持不变。"""
    if model_name not in POINT_CLOUD_MODEL_SPECS:
        raise ValueError(f"Unknown point_cloud_model={model_name}")
    name, default_config, default_checkpoint, url, default_min_bytes, label_map = (
        POINT_CLOUD_MODEL_SPECS[model_name]
    )
    custom_checkpoint = bool(checkpoint)
    return PointCloudModelSpec(
        name=name,
        config=config or default_config,
        checkpoint=checkpoint or default_checkpoint,
        checkpoint_url="" if custom_checkpoint else url,
        min_bytes=max(int(min_bytes), 1) if custom_checkpoint else int(default_min_bytes),
        label_map=tuple(label_map) if label_map is not None else None,
    )


def ensure_checkpoint(path: str, url: str, min_bytes: int) -> str:
    """确认 checkpoint 可用；缺失时下载，小于阈值时视为残缺。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) >= int(min_bytes):
        return path
    if os.path.exists(path) and os.path.getsize(path) < int(min_bytes):
        raise RuntimeError(
            f"Checkpoint exists but is too small: {path} "
            f"({os.path.getsize(path)} bytes < {min_bytes}). URL: {url}"
        )
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download checkpoint from {url} to {path}: {exc}") from exc
    if os.path.getsize(path) < int(min_bytes):
        raise RuntimeError(
            f"Downloaded checkpoint is incomplete: {path} "
            f"({os.path.getsize(path)} bytes < {min_bytes}). URL: {url}"
        )
    return path


def validate_local_checkpoint(path: str, min_bytes: int) -> str:
    """验证用户显式提供的本地 checkpoint，禁止静默回退到默认权重。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Custom checkpoint does not exist: {path}")
    size = os.path.getsize(path)
    if size < int(min_bytes):
        raise RuntimeError(
            f"Custom checkpoint is too small: {path} ({size} bytes < {min_bytes})"
        )
    return path


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _normalise_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _as_tensor(value: Any, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _rotate_camera_y(points: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angles).to(dtype=points.dtype)
    sine = torch.sin(angles).to(dtype=points.dtype)
    x = cosine * points[:, 0] + sine * points[:, 2]
    z = -sine * points[:, 0] + cosine * points[:, 2]
    return torch.stack([x, points[:, 1], z], dim=1)


def boxes2d_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """计算 ``(x1,y1,x2,y2)`` 图像框 IoU。"""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0)
        * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0)
    )[:, None]
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0)
        * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0)
    )[None, :]
    return inter / (area_a + area_b - inter).clamp_min(1e-6)


def camera_boxes_iou3d(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """计算 KITTI camera box 的旋转 3D IoU 矩阵。

    Args:
        boxes_a: ``(M,7)``，camera rect ``(x,y,z,w,l,h,ry)``，其中
            ``y`` 是框底面中心。
        boxes_b: ``(N,7)``，格式与 ``boxes_a`` 相同。

    Returns:
        ``(M,N)`` 旋转 3D IoU。BEV 使用 camera ``x/z`` 平面，高度交集
        使用 ``[y-h, y]``，避免把 camera 深度轴误当高度轴。
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    from mmcv.ops import box_iou_rotated

    bev_a = torch.stack(
        [boxes_a[:, 0], boxes_a[:, 2], boxes_a[:, 3], boxes_a[:, 4], boxes_a[:, 6]],
        dim=1,
    )
    bev_b = torch.stack(
        [boxes_b[:, 0], boxes_b[:, 2], boxes_b[:, 3], boxes_b[:, 4], boxes_b[:, 6]],
        dim=1,
    )
    bev_iou = box_iou_rotated(bev_a, bev_b).clamp(0.0, 1.0)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).clamp_min(0.0)[:, None]
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).clamp_min(0.0)[None, :]
    inter_area = bev_iou * (area_a + area_b) / (1.0 + bev_iou).clamp_min(1e-6)

    top_a = (boxes_a[:, 1] - boxes_a[:, 5])[:, None]
    bottom_a = boxes_a[:, 1][:, None]
    top_b = (boxes_b[:, 1] - boxes_b[:, 5])[None, :]
    bottom_b = boxes_b[:, 1][None, :]
    inter_h = (torch.minimum(bottom_a, bottom_b) - torch.maximum(top_a, top_b)).clamp_min(0.0)
    inter_volume = inter_area * inter_h
    volume_a = area_a * boxes_a[:, 5].clamp_min(0.0)[:, None]
    volume_b = area_b * boxes_b[:, 5].clamp_min(0.0)[None, :]
    return inter_volume / (volume_a + volume_b - inter_volume).clamp_min(1e-6)


class MMDetInstanceSegAdapter(nn.Module):
    """mmdetection Mask R-CNN 适配器。

    测试时从真实模型输出 ``boxes_2d (K,4)``、``masks_2d (K,H,W)``、
    ``labels (K,)``、``scores (K,)``。训练时若 batch 提供 KITTI GT 2D 框，
    优先使用 GT 框作为 FF-Net 监督来源，同时预训练模型仍实例化并冻结。
    """

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str,
        score_thr: float = 0.2,
        max_dets: int = 100,
        use_gt_boxes_in_train: bool = True,
    ):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet")
        self.model = init_detector(config, checkpoint, device=device)
        if hasattr(self.model, "cfg") and "test_pipeline" in self.model.cfg:
            self.model.cfg.test_pipeline = [
                step for step in self.model.cfg.test_pipeline
                if step.get("type") != "LoadAnnotations"
            ]
        if hasattr(self.model, "cfg") and "test_dataloader" in self.model.cfg:
            dataset = self.model.cfg.test_dataloader.get("dataset", None)
            if dataset is not None and "pipeline" in dataset:
                dataset.pipeline = [
                    step for step in dataset.pipeline
                    if step.get("type") != "LoadAnnotations"
                ]
        self.score_thr = float(score_thr)
        self.max_dets = int(max_dets)
        self.use_gt_boxes_in_train = bool(use_gt_boxes_in_train)
        freeze_module(self.model)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = batch["points"].device
        if self.use_gt_boxes_in_train and batch.get("mode") == "train" and "gt_boxes_2d" in batch:
            boxes = batch["gt_boxes_2d"].to(device=device, dtype=torch.float32)
            labels = batch["gt_labels"].to(device=device, dtype=torch.long)
            scores = torch.ones((boxes.shape[0],), dtype=torch.float32, device=device)
            image_shape = batch.get("image_shape", (0, 0))
            masks = torch.empty((0,), dtype=torch.float32, device=device)
            if boxes.numel() > 0 and image_shape[0] > 0 and image_shape[1] > 0:
                masks = torch.zeros((boxes.shape[0], int(image_shape[0]), int(image_shape[1])), device=device)
                for idx, box in enumerate(boxes.round().long()):
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, min(x1, int(image_shape[1]) - 1))
                    x2 = max(0, min(x2, int(image_shape[1])))
                    y1 = max(0, min(y1, int(image_shape[0]) - 1))
                    y2 = max(0, min(y2, int(image_shape[0])))
                    if x2 > x1 and y2 > y1:
                        masks[idx, y1:y2, x1:x2] = 1.0
            return {"boxes_2d": boxes, "masks_2d": masks, "labels": labels, "scores": scores}

        from mmdet.apis import inference_detector
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")
        result = inference_detector(self.model, batch["image_path"])
        pred = result.pred_instances
        scores = pred.scores.to(device=device)
        raw_labels = pred.labels.to(device=device, dtype=torch.long)
        class_keep = (raw_labels == 0) | (raw_labels == 1) | (raw_labels == 2)
        keep = (scores >= self.score_thr) & class_keep
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[: self.max_dets]
        boxes = pred.bboxes.to(device=device, dtype=torch.float32)[keep_idx]
        labels = raw_labels[keep_idx]
        scores = scores[keep_idx].to(dtype=torch.float32)
        if hasattr(pred, "masks") and pred.masks is not None:
            masks = pred.masks.to(device=device, dtype=torch.float32)[keep_idx]
        else:
            masks = torch.empty((0,), dtype=torch.float32, device=device)
        return {"boxes_2d": boxes, "masks_2d": masks, "labels": labels, "scores": scores}


class MMDet3DNearDetectorAdapter(nn.Module):
    """mmdet3d 近距离 detector 适配器，输出 camera rect 坐标 3D box。"""

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str,
        score_thr: float = 0.05,
        label_map: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet3d.apis import init_model
        from mmdet3d.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet3d")
        self.model = init_model(config, checkpoint, device=device)
        self.score_thr = float(score_thr)
        self.label_map = tuple(int(x) for x in label_map) if label_map is not None else None
        freeze_module(self.model)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from mmdet3d.apis import inference_detector
        from mmdet3d.structures import Box3DMode
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet3d")
        device = batch["points"].device
        result = inference_detector(self.model, batch["points_path"])
        if isinstance(result, tuple):
            result = result[0]
        pred = result.pred_instances_3d
        scores = pred.scores_3d.to(device=device, dtype=torch.float32)
        keep = scores >= self.score_thr
        if keep.numel() == 0:
            return {
                "boxes_3d_camera": torch.zeros((0, 7), dtype=torch.float32, device=device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            }
        p2 = batch["calib"]["P2"]
        del p2
        r0 = _as_tensor(batch["calib"]["R0_rect"], str(device))
        tr = _as_tensor(batch["calib"]["Tr_velo_to_cam"], str(device))
        rt = torch.eye(4, dtype=torch.float32, device=device)
        rt[:3, :4] = r0 @ tr
        boxes_cam = pred.bboxes_3d.convert_to(Box3DMode.CAM, rt_mat=rt.detach().cpu().numpy(), correct_yaw=True)
        tensor = boxes_cam.tensor.to(device=device, dtype=torch.float32)[keep]
        labels = pred.labels_3d.to(device=device, dtype=torch.long)[keep]
        scores = scores[keep]
        if self.label_map is not None:
            label_map = torch.tensor(self.label_map, dtype=torch.long, device=device)
            valid_labels = (labels >= 0) & (labels < label_map.numel())
            tensor = tensor[valid_labels]
            scores = scores[valid_labels]
            labels = label_map[labels[valid_labels]]
        converted = torch.stack(
            [
                tensor[:, 0],
                tensor[:, 1],
                tensor[:, 2],
                tensor[:, 5],
                tensor[:, 3],
                tensor[:, 4],
                _normalise_angle(tensor[:, 6]),
            ],
            dim=1,
        )
        return {"boxes_3d_camera": converted, "scores": scores, "labels": labels}


@dataclass
class FarawayFrustumSystem:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    postprocessor: Any
    image_adapter: nn.Module
    point_cloud_adapter: nn.Module
    device: str
    class_names: Tuple[str, ...]


class FarawayFrustumCriterion(nn.Module):
    """Faraway FF-Net 监督损失。

    使用 KITTI camera box GT 监督远距离分支输出，不对冻结 2D/3D detector 反传。
    训练时优先使用 framework 传入的 ``source_indices`` 做 GT 2D 框到 Faraway
    预测的一一对应；缺失时才回退到同类别 x/z 最近邻匹配。
    """

    def __init__(
        self,
        depth_thresholds: Optional[Iterable[float]] = None,
        train_class_ids: Optional[Iterable[int]] = None,
        match_radius: float = 25.0,
        depth_weight: float = 1.0,
        size_weight: float = 1.0,
        yaw_weight: float = 0.35,
        score_weight: float = 0.5,
        bb_shift: float = 75.0,
        match_2d_iou_thr: float = 0.5,
        score_positive_radius: float = 4.0,
        score_raw_miss_radius: float = 0.0,
        near_score_weight: float = 0.0,
        near_preserve_weight: float = 0.0,
        near_rank_weight: float = 0.0,
        near_rank_margin: float = 0.5,
        near_negative_only_weight: float = 1.0,
        near_positive_iou_thr: float = 0.70,
        near_negative_iou_thr: float = 0.30,
        near_score_only: bool = False,
    ):
        super().__init__()
        if depth_thresholds is None:
            depth_thresholds = [60.0, 60.0, 75.0]
        self.register_buffer("depth_thresholds", torch.tensor(list(depth_thresholds), dtype=torch.float32))
        class_ids = list(train_class_ids) if train_class_ids is not None else []
        self.register_buffer("train_class_ids", torch.tensor(class_ids, dtype=torch.long))
        self.match_radius = float(match_radius)
        self.depth_weight = float(depth_weight)
        self.size_weight = float(size_weight)
        self.yaw_weight = float(yaw_weight)
        self.score_weight = float(score_weight)
        self.bb_shift = float(bb_shift)
        self.match_2d_iou_thr = float(match_2d_iou_thr)
        self.score_positive_radius = float(score_positive_radius)
        self.score_raw_miss_radius = float(score_raw_miss_radius)
        self.near_score_weight = float(near_score_weight)
        self.near_preserve_weight = float(near_preserve_weight)
        self.near_rank_weight = float(near_rank_weight)
        self.near_rank_margin = float(near_rank_margin)
        self.near_negative_only_weight = float(near_negative_only_weight)
        self.near_positive_iou_thr = float(near_positive_iou_thr)
        self.near_negative_iou_thr = float(near_negative_iou_thr)
        self.near_score_only = bool(near_score_only)
        if self.near_negative_iou_thr >= self.near_positive_iou_thr:
            raise ValueError("near_negative_iou_thr must be smaller than near_positive_iou_thr")
        if self.near_rank_weight < 0.0 or self.near_negative_only_weight < 0.0:
            raise ValueError("near ranking weights must be non-negative")

    def _empty_loss(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        anchor = pred["faraway"].get("regression_outputs", pred["boxes_3d_camera"]).sum() * 0.0
        zero_long = torch.tensor(0, device=anchor.device, dtype=torch.long)
        zero_float = torch.tensor(0.0, device=anchor.device)
        return {
            "loss_total": anchor,
            "loss_box": anchor,
            "loss_center": anchor,
            "loss_depth": anchor,
            "loss_y": anchor,
            "loss_size": anchor,
            "loss_yaw": anchor,
            "loss_score": anchor,
            "num_faraway": zero_long,
            "num_matched": zero_long,
            "mean_match_distance": zero_float,
            "mean_score_target": zero_float,
        }

    def _class_allowed(self, labels: torch.Tensor) -> torch.Tensor:
        if self.train_class_ids.numel() == 0:
            return torch.ones_like(labels, dtype=torch.bool)
        allowed = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.train_class_ids.to(device=labels.device):
            allowed = allowed | (labels == class_id)
        return allowed

    def _add_near_losses(
        self,
        losses: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        near = pred.get("near", {})
        logits = near.get("calibration_logits")
        residual = near.get("calibration_residual")
        boxes = near.get("boxes_3d_camera")
        labels = near.get("labels")
        anchor = losses["loss_total"]
        zero = anchor * 0.0
        zero_long = torch.tensor(0, device=anchor.device, dtype=torch.long)
        zero_float = torch.tensor(0.0, device=anchor.device)
        near_losses = {
            "loss_near_score": zero,
            "loss_near_rank": zero,
            "loss_near_preserve": zero,
            "num_near_positive": zero_long,
            "num_near_negative": zero_long,
            "num_near_ignored": zero_long,
            "mean_near_max_iou": zero_float,
        }
        if (
            logits is None
            or residual is None
            or boxes is None
            or labels is None
            or logits.numel() == 0
            or logits.shape[0] != boxes.shape[0]
        ):
            losses.update(near_losses)
            return losses

        boxes = boxes.to(device=logits.device, dtype=logits.dtype)
        labels = labels.to(device=logits.device, dtype=torch.long)
        gt_boxes = batch.get("gt_boxes_3d_camera")
        gt_labels = batch.get("gt_labels")
        if gt_boxes is None or gt_labels is None:
            losses.update(near_losses)
            return losses
        gt_boxes = gt_boxes.to(device=logits.device, dtype=logits.dtype)
        gt_labels = gt_labels.to(device=logits.device, dtype=torch.long)
        if gt_boxes.numel() == 0:
            max_iou = logits.new_zeros((logits.shape[0],))
        else:
            iou_matrix = camera_boxes_iou3d(boxes.detach(), gt_boxes.detach())
            same_class = labels[:, None] == gt_labels[None, :]
            iou_matrix = iou_matrix * same_class.to(dtype=iou_matrix.dtype)
            max_iou = iou_matrix.amax(dim=1)

        allowed = self._class_allowed(labels)
        positive = allowed & (max_iou >= self.near_positive_iou_thr)
        negative = allowed & (max_iou <= self.near_negative_iou_thr)
        valid = positive | negative
        if bool(valid.any().item()):
            targets = torch.where(positive[valid], max_iou[valid], torch.zeros_like(max_iou[valid]))
            num_positive = int(positive.sum().item())
            num_negative = int(negative.sum().item())
            positive_weight = max(
                1.0,
                min(float(num_negative) / max(float(num_positive), 1.0), 10.0),
            )
            sample_weights = torch.where(
                positive[valid],
                torch.full_like(targets, positive_weight),
                torch.ones_like(targets),
            )
            loss_near_score = (
                F.binary_cross_entropy_with_logits(logits[valid], targets, reduction="none")
                * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)
            if num_positive == 0:
                loss_near_score = loss_near_score * self.near_negative_only_weight
        else:
            loss_near_score = logits.sum() * 0.0
            num_positive = 0
            num_negative = 0
        if bool(positive.any().item()) and bool(negative.any().item()):
            positive_logits = logits[positive]
            negative_logits = logits[negative]
            pairwise_gap = positive_logits[:, None] - negative_logits[None, :]
            loss_near_rank = F.softplus(self.near_rank_margin - pairwise_gap).mean()
        else:
            loss_near_rank = logits.sum() * 0.0
        loss_near_preserve = F.smooth_l1_loss(residual, torch.zeros_like(residual))
        losses["loss_total"] = (
            losses["loss_total"]
            + self.near_score_weight * loss_near_score
            + self.near_rank_weight * loss_near_rank
            + self.near_preserve_weight * loss_near_preserve
        )
        near_losses.update(
            {
                "loss_near_score": loss_near_score,
                "loss_near_rank": loss_near_rank,
                "loss_near_preserve": loss_near_preserve,
                "num_near_positive": torch.tensor(num_positive, device=logits.device, dtype=torch.long),
                "num_near_negative": torch.tensor(num_negative, device=logits.device, dtype=torch.long),
                "num_near_ignored": torch.tensor(
                    int((allowed & ~valid).sum().item()), device=logits.device, dtype=torch.long
                ),
                "mean_near_max_iou": max_iou[allowed].detach().mean() if bool(allowed.any().item()) else zero_float,
            }
        )
        losses.update(near_losses)
        return losses

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.near_score_only:
            return self._add_near_losses(self._empty_loss(pred), pred, batch)
        device = pred["boxes_3d_camera"].device
        far_boxes = pred["faraway"]["boxes_3d_camera"]
        far_labels = pred["faraway"].get("labels")
        far_regression = pred["faraway"].get("regression_outputs")
        far_centroids = pred["faraway"].get("aligned_centroids", pred["faraway"].get("centroids"))
        far_match_centroids = pred["faraway"].get("centroids")
        frustum_angles = pred["faraway"].get("frustum_angles")
        source_boxes_2d = pred["faraway"].get("source_boxes_2d")
        source_indices = pred["faraway"].get("source_indices")
        gt_boxes = batch.get("gt_boxes_3d_camera")
        gt_boxes_2d = batch.get("gt_boxes_2d")
        gt_labels = batch.get("gt_labels")
        score_logits = pred["faraway"].get("score_logits")
        if gt_boxes is None or gt_labels is None:
            losses = self._empty_loss(pred)
            if score_logits is not None and score_logits.numel() > 0:
                loss_score = F.binary_cross_entropy_with_logits(
                    score_logits,
                    torch.zeros_like(score_logits),
                )
                losses["loss_score"] = loss_score
                losses["loss_total"] = self.score_weight * loss_score
            return self._add_near_losses(losses, pred, batch)
        gt_boxes = gt_boxes.to(device=device, dtype=far_boxes.dtype)
        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        if far_labels is None:
            far_labels = torch.zeros((far_boxes.shape[0],), dtype=torch.long, device=device)
        else:
            far_labels = far_labels.to(device=device, dtype=torch.long)
        if far_boxes.numel() == 0:
            return self._add_near_losses(self._empty_loss(pred), pred, batch)

        far_allowed = self._class_allowed(far_labels)
        has_source_boxes_2d = source_boxes_2d is not None and gt_boxes_2d is not None
        has_source_indices = source_indices is not None
        if gt_boxes.numel() == 0:
            direct_valid = torch.zeros((far_boxes.shape[0],), dtype=torch.bool, device=device)
            fallback_target = far_boxes.new_zeros((far_boxes.shape[0], 7))
            fallback_dist = far_boxes.new_full((far_boxes.shape[0],), float("inf"))
        elif has_source_boxes_2d:
            source_boxes_2d = source_boxes_2d.to(device=device, dtype=far_boxes.dtype)
            gt_boxes_2d = gt_boxes_2d.to(device=device, dtype=far_boxes.dtype)
            ious = boxes2d_iou(source_boxes_2d, gt_boxes_2d)
            if ious.numel() > 0:
                best_iou, best_gt = ious.max(dim=1)
            else:
                best_iou = far_boxes.new_zeros((far_boxes.shape[0],))
                best_gt = torch.zeros((far_boxes.shape[0],), dtype=torch.long, device=device)
            direct_target = gt_boxes[best_gt.clamp(min=0, max=max(gt_boxes.shape[0] - 1, 0))]
            direct_labels = gt_labels[best_gt.clamp(min=0, max=max(gt_labels.shape[0] - 1, 0))]
            match_centers = far_boxes
            if (
                far_match_centroids is not None
                and far_match_centroids.dim() == 2
                and far_match_centroids.shape[0] == far_boxes.shape[0]
            ):
                match_centers = far_match_centroids.to(device=device, dtype=far_boxes.dtype)
            direct_dist = torch.norm(match_centers[:, [0, 2]] - direct_target[:, [0, 2]], dim=1)
            direct_valid = (
                far_allowed
                & (direct_labels == far_labels)
                & (best_iou >= self.match_2d_iou_thr)
                & (direct_dist <= self.match_radius)
            )
            fallback_target = direct_target.clone()
            fallback_dist = direct_dist.clone()
        elif has_source_indices:
            source_indices = source_indices.to(device=device, dtype=torch.long)
            direct_valid = (
                (source_indices >= 0)
                & (source_indices < gt_boxes.shape[0])
                & far_allowed
            )
            direct_target = gt_boxes[source_indices.clamp(min=0, max=max(gt_boxes.shape[0] - 1, 0))]
            direct_labels = gt_labels[source_indices.clamp(min=0, max=max(gt_labels.shape[0] - 1, 0))]
            direct_valid = direct_valid & (direct_labels == far_labels)
            match_centers = far_boxes
            if (
                far_match_centroids is not None
                and far_match_centroids.dim() == 2
                and far_match_centroids.shape[0] == far_boxes.shape[0]
            ):
                match_centers = far_match_centroids.to(device=device, dtype=far_boxes.dtype)
            direct_dist = torch.norm(match_centers[:, [0, 2]] - direct_target[:, [0, 2]], dim=1)
            direct_valid = direct_valid & (direct_dist <= self.match_radius)
            fallback_target = direct_target.clone()
            fallback_dist = direct_dist.clone()
        else:
            direct_valid = torch.zeros((far_boxes.shape[0],), dtype=torch.bool, device=device)
            direct_target = far_boxes.new_zeros((far_boxes.shape[0], 7))
            direct_dist = far_boxes.new_full((far_boxes.shape[0],), float("inf"))
            fallback_target = direct_target.clone()
            fallback_dist = direct_dist.clone()

        needs_fallback = (~direct_valid) & far_allowed & (not has_source_indices) & (not has_source_boxes_2d)
        if gt_boxes.numel() > 0 and bool(needs_fallback.any().item()):
            for pred_idx in torch.nonzero(needs_fallback, as_tuple=False).squeeze(1):
                same_class = gt_labels == far_labels[pred_idx]
                same_class = same_class & self._class_allowed(gt_labels)
                if not bool(same_class.any().item()):
                    continue
                candidates = gt_boxes[same_class]
                candidate_dist = torch.norm(candidates[:, [0, 2]] - far_boxes[pred_idx, [0, 2]], dim=1)
                best = torch.argmin(candidate_dist)
                if float(candidate_dist[best].detach().cpu()) <= self.match_radius:
                    fallback_target[pred_idx] = candidates[best]
                    fallback_dist[pred_idx] = candidate_dist[best]
                    direct_valid[pred_idx] = True

        matched = direct_valid
        score_quality_valid = matched
        if self.score_positive_radius > 0.0:
            score_quality_valid = matched & (fallback_dist <= self.score_positive_radius)
        score_targets = far_boxes.new_zeros((far_boxes.shape[0],))
        if bool(score_quality_valid.any().item()):
            quality_indices = torch.nonzero(score_quality_valid, as_tuple=False).squeeze(1)
            paired_iou = camera_boxes_iou3d(
                far_boxes[quality_indices].detach(),
                fallback_target[quality_indices].detach(),
            ).diag().clamp(0.0, 1.0)
            score_targets[quality_indices] = paired_iou
        raw_miss = torch.ones_like(matched, dtype=torch.bool)
        if self.score_raw_miss_radius > 0.0 and "near" in pred:
            near = pred["near"]
            near_boxes = near.get("boxes_3d_camera")
            near_labels = near.get("labels")
            if near_boxes is not None and near_labels is not None and near_boxes.numel() > 0:
                near_boxes = near_boxes.to(device=device, dtype=far_boxes.dtype)
                near_labels = near_labels.to(device=device, dtype=torch.long)
                for pred_idx in torch.nonzero(matched, as_tuple=False).squeeze(1):
                    same_class = near_labels == far_labels[pred_idx]
                    if not bool(same_class.any().item()):
                        continue
                    near_dist = torch.norm(
                        near_boxes[same_class][:, [0, 2]] - far_boxes[pred_idx, [0, 2]],
                        dim=1,
                    )
                    raw_miss[pred_idx] = ~(near_dist < self.score_raw_miss_radius).any()
                score_targets = score_targets * raw_miss.to(dtype=score_targets.dtype)

        if score_logits is not None and score_logits.numel() == far_boxes.shape[0]:
            score_mask = far_allowed
            if bool(score_mask.any().item()):
                score_loss_values = F.binary_cross_entropy_with_logits(
                    score_logits[score_mask],
                    score_targets.to(dtype=score_logits.dtype)[score_mask],
                    reduction="none",
                )
                score_weights = 1.0 + 4.0 * score_targets[score_mask]
                loss_score = (score_loss_values * score_weights).sum() / score_weights.sum().clamp_min(1.0)
            else:
                loss_score = score_logits.sum() * 0.0
        else:
            loss_score = far_boxes.sum() * 0.0

        if not bool(matched.any().item()):
            losses = self._empty_loss(pred)
            losses["num_faraway"] = torch.tensor(far_boxes.shape[0], device=device, dtype=torch.long)
            losses["loss_score"] = loss_score
            losses["loss_total"] = self.score_weight * loss_score
            return self._add_near_losses(losses, pred, batch)

        matched_indices = torch.nonzero(matched, as_tuple=False).squeeze(1)
        far_boxes = far_boxes[matched]
        target = fallback_target[matched]
        match_dist = fallback_dist[matched]
        if (
            far_regression is not None
            and far_regression.dim() == 2
            and far_regression.shape[1] == 7
            and far_centroids is not None
        ):
            reg = far_regression.to(device=device, dtype=far_boxes.dtype)[matched_indices]
            centroids = far_centroids.to(device=device, dtype=far_boxes.dtype)[matched_indices]
            if frustum_angles is not None:
                matched_angles = frustum_angles.to(device=device, dtype=far_boxes.dtype)[matched_indices]
                target_center = _rotate_camera_y(target[:, 0:3], -matched_angles)
                target_yaw = _normalise_angle(target[:, 6] - matched_angles)
            else:
                target_center = target[:, 0:3]
                target_yaw = _normalise_angle(target[:, 6])
            target_reg = torch.stack(
                [
                    target_center[:, 0] + self.bb_shift,
                    target_center[:, 1],
                    target_center[:, 2] + self.bb_shift - centroids[:, 2],
                    target[:, 5],
                    target[:, 3],
                    target[:, 4],
                    target_yaw,
                ],
                dim=1,
            )
            loss_center = F.smooth_l1_loss(reg[:, 0:3], target_reg[:, 0:3])
            loss_depth = F.smooth_l1_loss(reg[:, 2], target_reg[:, 2])
            loss_y = F.smooth_l1_loss(reg[:, 1], target_reg[:, 1])
            loss_size = F.smooth_l1_loss(
                torch.log(reg[:, 3:6].clamp_min(1e-3)),
                torch.log(target_reg[:, 3:6].clamp_min(1e-3)),
            )
            yaw_delta = 0.5 * torch.atan2(
                torch.sin(2.0 * (reg[:, 6] - target_reg[:, 6])),
                torch.cos(2.0 * (reg[:, 6] - target_reg[:, 6])),
            )
            loss_yaw = F.smooth_l1_loss(yaw_delta, torch.zeros_like(yaw_delta))
        else:
            pred_vec = torch.stack(
                [
                    far_boxes[:, 0],
                    far_boxes[:, 1],
                    far_boxes[:, 2],
                    far_boxes[:, 3],
                    far_boxes[:, 4],
                    far_boxes[:, 5],
                    _normalise_angle(far_boxes[:, 6]),
                ],
                dim=1,
            )
            target_vec = torch.stack(
                [
                    target[:, 0],
                    target[:, 1],
                    target[:, 2],
                    target[:, 3],
                    target[:, 4],
                    target[:, 5],
                    _normalise_angle(target[:, 6]),
                ],
                dim=1,
            )
            loss_center = F.smooth_l1_loss(pred_vec[:, 0:3], target_vec[:, 0:3])
            loss_depth = F.smooth_l1_loss(pred_vec[:, 2], target_vec[:, 2])
            loss_y = F.smooth_l1_loss(pred_vec[:, 1], target_vec[:, 1])
            loss_size = F.smooth_l1_loss(
                torch.log(pred_vec[:, 3:6].clamp_min(1e-3)),
                torch.log(target_vec[:, 3:6].clamp_min(1e-3)),
            )
            yaw_delta = 0.5 * torch.atan2(
                torch.sin(2.0 * (pred_vec[:, 6] - target_vec[:, 6])),
                torch.cos(2.0 * (pred_vec[:, 6] - target_vec[:, 6])),
            )
            loss_yaw = F.smooth_l1_loss(yaw_delta, torch.zeros_like(yaw_delta))
        loss_box = self.depth_weight * loss_center + self.size_weight * loss_size + self.yaw_weight * loss_yaw
        loss_total = loss_box + self.score_weight * loss_score
        losses = {
            "loss_total": loss_total,
            "loss_box": loss_box,
            "loss_center": loss_center,
            "loss_depth": loss_depth,
            "loss_y": loss_y,
            "loss_size": loss_size,
            "loss_yaw": loss_yaw,
            "loss_score": loss_score,
            "num_faraway": torch.tensor(pred["faraway"]["boxes_3d_camera"].shape[0], device=device, dtype=torch.long),
            "num_matched": torch.tensor(far_boxes.shape[0], device=device, dtype=torch.long),
            "num_score_positive": torch.tensor(int((score_targets > 0.5).sum().item()), device=device, dtype=torch.long),
            "mean_match_distance": match_dist.detach().mean(),
            "mean_score_target": score_targets[far_allowed].detach().mean()
            if bool(far_allowed.any().item())
            else far_boxes.new_tensor(0.0),
        }
        return self._add_near_losses(losses, pred, batch)


def boxes3d_camera_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """camera rect box ``(x,y,z,w,l,h,ry)`` -> corners ``(N,8,3)``。"""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8, 3))
    x, y, z, w, l, h, ry = boxes.unbind(dim=1)
    x_c = torch.stack([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dim=1)
    y_c = torch.stack([torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h), -h, -h, -h, -h], dim=1)
    z_c = torch.stack([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dim=1)
    cos_r = torch.cos(ry)
    sin_r = torch.sin(ry)
    corner_x = cos_r[:, None] * x_c + sin_r[:, None] * z_c + x[:, None]
    corner_y = y_c + y[:, None]
    corner_z = -sin_r[:, None] * x_c + cos_r[:, None] * z_c + z[:, None]
    return torch.stack([corner_x, corner_y, corner_z], dim=2)


class FarawayFrustumPostProcessor:
    """KITTI label_2 输出与 evaluator 调用辅助。"""

    def __init__(
        self,
        class_names: Sequence[str] = KITTI_CLASS_NAMES,
        faraway_score_thr: float = 0.0,
        faraway_class_ids: Optional[Iterable[int]] = None,
        faraway_near_suppress_radius: float = 0.0,
        paper_distance_fusion: bool = False,
        pedestrian_far_distance: float = 60.0,
        car_far_distance: float = 75.0,
        faraway_nms_radius: float = 0.0,
        faraway_topk_per_class: int = 0,
        car_score_thr: float = 0.0,
    ):
        self.class_names = tuple(class_names)
        self.faraway_score_thr = float(faraway_score_thr)
        self.faraway_class_ids = tuple(int(x) for x in faraway_class_ids) if faraway_class_ids is not None else tuple()
        self.faraway_near_suppress_radius = float(faraway_near_suppress_radius)
        self.paper_distance_fusion = bool(paper_distance_fusion)
        self.pedestrian_far_distance = float(pedestrian_far_distance)
        self.car_far_distance = float(car_far_distance)
        self.faraway_nms_radius = float(faraway_nms_radius)
        self.faraway_topk_per_class = int(faraway_topk_per_class)
        self.car_score_thr = float(car_score_thr)

    def _paper_distance_threshold(self, label_id: int) -> Optional[float]:
        if label_id < 0 or label_id >= len(self.class_names):
            return None
        class_name = self.class_names[label_id]
        if class_name == "Pedestrian":
            return self.pedestrian_far_distance
        if class_name == "Car":
            return self.car_far_distance
        return None

    def _select_faraway_indices(self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, is_faraway: torch.Tensor) -> torch.Tensor:
        """按类别对 Faraway 候选做 BEV 距离去重和 Top-K 保留。"""
        keep = torch.ones((boxes.shape[0],), dtype=torch.bool)
        far_indices = torch.nonzero(is_faraway, as_tuple=False).squeeze(1)
        if far_indices.numel() == 0:
            return keep
        for label_id in labels[far_indices].unique():
            class_indices = far_indices[labels[far_indices] == label_id]
            if class_indices.numel() == 0:
                continue
            order = class_indices[torch.argsort(scores[class_indices], descending=True)]
            selected: List[int] = []
            for idx_tensor in order:
                idx = int(idx_tensor.item())
                if self.faraway_topk_per_class > 0 and len(selected) >= self.faraway_topk_per_class:
                    keep[idx] = False
                    continue
                if self.faraway_nms_radius > 0.0 and selected:
                    selected_tensor = torch.tensor(selected, dtype=torch.long)
                    distances = torch.norm(boxes[selected_tensor][:, [0, 2]] - boxes[idx, [0, 2]], dim=1)
                    if bool((distances < self.faraway_nms_radius).any().item()):
                        keep[idx] = False
                        continue
                selected.append(idx)
        return keep

    def project_boxes_to_2d(
        self,
        boxes_3d_camera: torch.Tensor,
        p2: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        corners = boxes3d_camera_to_corners(boxes_3d_camera)
        if corners.numel() == 0:
            return boxes_3d_camera.new_zeros((0, 4))
        ones = torch.ones((*corners.shape[:2], 1), dtype=corners.dtype, device=corners.device)
        corners_h = torch.cat([corners, ones], dim=2)
        proj = torch.matmul(corners_h, p2.to(device=corners.device, dtype=corners.dtype).t())
        uv = proj[..., :2] / proj[..., 2:3].clamp(min=1e-6)
        x1 = uv[..., 0].amin(dim=1).clamp(0, float(image_shape[1] - 1))
        y1 = uv[..., 1].amin(dim=1).clamp(0, float(image_shape[0] - 1))
        x2 = uv[..., 0].amax(dim=1).clamp(0, float(image_shape[1] - 1))
        y2 = uv[..., 1].amax(dim=1).clamp(0, float(image_shape[0] - 1))
        return torch.stack([x1, y1, x2, y2], dim=1)

    def write_kitti_prediction(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, Any], pred_dir: str) -> str:
        os.makedirs(pred_dir, exist_ok=True)
        sample_id = str(batch["sample_id"])
        out_path = os.path.join(pred_dir, f"{sample_id}.txt")
        boxes = predictions["boxes_3d_camera"].detach().float().cpu()
        scores = predictions["scores"].detach().float().cpu()
        labels = predictions["labels"].detach().long().cpu()
        is_faraway = predictions.get("is_faraway_prediction")
        if is_faraway is None:
            is_faraway = torch.zeros((boxes.shape[0],), dtype=torch.bool)
        else:
            is_faraway = is_faraway.detach().bool().cpu()
        p2 = _as_tensor(batch["calib"]["P2"], "cpu")
        corners = boxes3d_camera_to_corners(boxes)
        valid_depth = (boxes[:, 2] > 0.1) & (corners[..., 2].amin(dim=1) > 0.1)
        valid_dims = torch.isfinite(boxes).all(dim=1) & (boxes[:, 3:6] > 0).all(dim=1)
        valid_scores = torch.isfinite(scores) & (scores > 0)
        valid_mask = valid_depth & valid_dims & valid_scores
        faraway_keep = self._select_faraway_indices(boxes, scores, labels, is_faraway)
        bboxes_2d = self.project_boxes_to_2d(boxes, p2, batch["image_shape"]).cpu()
        lines: List[str] = []
        for idx in range(boxes.shape[0]):
            if not bool(valid_mask[idx].item()):
                continue
            if not bool(faraway_keep[idx].item()):
                continue
            label_id = int(labels[idx].item())
            if label_id < 0 or label_id >= len(self.class_names):
                continue
            if (
                self.class_names[label_id] == "Car"
                and not bool(is_faraway[idx].item())
                and scores[idx].item() < self.car_score_thr
            ):
                continue
            distance_thr = self._paper_distance_threshold(label_id)
            if self.paper_distance_fusion and distance_thr is not None:
                is_far = bool(is_faraway[idx].item())
                z_distance = float(boxes[idx, 2].item())
                if is_far and z_distance < distance_thr:
                    continue
                if (not is_far) and z_distance > distance_thr:
                    continue
            if bool(is_faraway[idx].item()):
                if self.faraway_class_ids and label_id not in self.faraway_class_ids:
                    continue
                if scores[idx].item() < self.faraway_score_thr:
                    continue
                if self.faraway_near_suppress_radius > 0.0:
                    near_mask = (~is_faraway) & (labels == label_id)
                    if self.paper_distance_fusion:
                        near_distance_thr = self._paper_distance_threshold(label_id)
                        if near_distance_thr is not None:
                            near_mask = near_mask & (boxes[:, 2] <= near_distance_thr)
                    if self.class_names[label_id] == "Car" and self.car_score_thr > 0.0:
                        near_mask = near_mask & (scores >= self.car_score_thr)
                    if bool(near_mask.any().item()):
                        near_boxes = boxes[near_mask]
                        near_dist = torch.norm(near_boxes[:, [0, 2]] - boxes[idx, [0, 2]], dim=1)
                        if bool((near_dist < self.faraway_near_suppress_radius).any().item()):
                            continue
            x, y, z, w, l, h, ry = boxes[idx].tolist()
            alpha = math.atan2(math.sin(ry - math.atan2(x, z)), math.cos(ry - math.atan2(x, z)))
            x1, y1, x2, y2 = bboxes_2d[idx].tolist()
            if abs(x2 - x1) < 1e-3 or abs(y2 - y1) < 1e-3:
                continue
            lines.append(
                f"{self.class_names[label_id]} 0.00 0 {alpha:.6f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.6f} {scores[idx].item():.6f}"
            )
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if lines:
                handle.write("\n")
        return out_path

    def evaluate(
        self,
        pred_dir: str,
        gt_annos: List[Dict[str, Any]],
        sample_ids: List[str],
        kitti_root: Optional[str] = None,
        native_eval: str = "kitti_native_evaluation/build/evaluate_object_3d_offline",
    ) -> str:
        if kitti_root is not None and os.path.isfile(native_eval):
            label_dir = os.path.join(kitti_root, "training", "label_2")
            result_root = os.path.dirname(pred_dir) if os.path.basename(pred_dir) == "data" else pred_dir
            completed = subprocess.run(
                [native_eval, label_dir, result_root],
                check=True,
                capture_output=True,
                text=True,
            )
            stats = []
            for name in sorted(os.listdir(result_root)):
                if name.startswith("stats_") and name.endswith(".txt"):
                    stats.append(name)
            return completed.stdout.strip() + ("\nstats_files=" + ",".join(stats) if stats else "")

        from mmdet3d.evaluation.functional import kitti_eval

        pred_annos = []
        for sample_id in sample_ids:
            pred_annos.append(read_kitti_prediction_as_anno(os.path.join(pred_dir, f"{sample_id}.txt")))
        result, _ = kitti_eval(gt_annos, pred_annos, self.class_names)
        return str(result)


def read_kitti_prediction_as_anno(path: str) -> Dict[str, np.ndarray]:
    names, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = [], [], [], [], [], [], [], [], []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 16:
                    continue
                names.append(parts[0])
                truncated.append(float(parts[1]))
                occluded.append(int(parts[2]))
                alpha.append(float(parts[3]))
                bbox.append([float(x) for x in parts[4:8]])
                dimensions.append([float(x) for x in parts[8:11]])
                location.append([float(x) for x in parts[11:14]])
                rotation_y.append(float(parts[14]))
                score.append(float(parts[15]))
    return {
        "name": np.array(names),
        "truncated": np.array(truncated, dtype=np.float32),
        "occluded": np.array(occluded, dtype=np.int64),
        "alpha": np.array(alpha, dtype=np.float32),
        "bbox": np.array(bbox, dtype=np.float32).reshape(-1, 4),
        "dimensions": np.array(dimensions, dtype=np.float32).reshape(-1, 3),
        "location": np.array(location, dtype=np.float32).reshape(-1, 3),
        "rotation_y": np.array(rotation_y, dtype=np.float32),
        "score": np.array(score, dtype=np.float32),
    }


def verify_frozen_modules(system: FarawayFrustumSystem) -> Dict[str, int]:
    image_trainable = sum(p.numel() for p in system.image_adapter.parameters() if p.requires_grad)
    point_cloud_trainable = sum(p.numel() for p in system.point_cloud_adapter.parameters() if p.requires_grad)
    optimizer_param_count = sum(p.numel() for group in system.optimizer.param_groups for p in group["params"])
    if image_trainable != 0:
        raise AssertionError(f"image_trainable must be 0, got {image_trainable}")
    if point_cloud_trainable != 0:
        raise AssertionError(f"point_cloud_trainable must be 0, got {point_cloud_trainable}")
    if optimizer_param_count <= 0:
        raise AssertionError("optimizer must contain trainable Faraway modules")
    return {
        "image_trainable": int(image_trainable),
        "point_cloud_trainable": int(point_cloud_trainable),
        "optimizer_param_count": int(optimizer_param_count),
    }


def build_faraway_frustum_system(
    device: str = "cuda:0",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    depth_thresholds: Optional[Iterable[float]] = None,
    image_score_thr: float = 0.2,
    max_2d_dets: int = 100,
    bev_width_meters: float = 20.0,
    bev_depth_meters: float = 30.0,
    bev_resolution: float = 0.1,
    faraway_score_mode: str = "image",
    faraway_image_geometry_alignment: bool = False,
    train_class_ids: Optional[Iterable[int]] = None,
    faraway_infer_class_ids: Optional[Iterable[int]] = None,
    faraway_score_thr: float = 0.0,
    faraway_near_suppress_radius: float = 0.0,
    criterion_match_radius: float = 25.0,
    yaw_loss_weight: float = 0.35,
    score_loss_weight: float = 0.5,
    train_with_detector_2d: bool = False,
    match_2d_iou_thr: float = 0.5,
    score_positive_radius: float = 4.0,
    score_raw_miss_radius: float = 0.0,
    paper_distance_fusion: bool = False,
    pedestrian_far_distance: float = 60.0,
    car_far_distance: float = 75.0,
    faraway_nms_radius: float = 0.0,
    faraway_topk_per_class: int = 0,
    car_score_thr: float = 0.0,
    point_cloud_model: str = "pv_rcnn_3class",
    point_cloud_config: str = "",
    point_cloud_checkpoint: str = "",
    point_cloud_checkpoint_min_bytes: int = 10 * 1024 * 1024,
    near_image_score_fusion: bool = False,
    near_image_score_power: float = 0.5,
    near_image_score_min_factor: float = 0.65,
    near_image_score_trainable: bool = False,
    near_image_score_residual_scale: float = 0.5,
    near_score_loss_weight: float = 0.0,
    near_score_preserve_weight: float = 0.0,
    near_rank_loss_weight: float = 0.0,
    near_rank_margin: float = 0.5,
    near_negative_only_weight: float = 1.0,
    near_positive_iou_thr: float = 0.70,
    near_negative_iou_thr: float = 0.30,
    train_near_score_only: bool = False,
    ensure_weights: bool = True,
) -> FarawayFrustumSystem:
    if train_near_score_only and not near_image_score_trainable:
        raise ValueError("train_near_score_only requires near_image_score_trainable")
    point_cloud_spec = resolve_point_cloud_spec(
        point_cloud_model,
        config=point_cloud_config,
        checkpoint=point_cloud_checkpoint,
        min_bytes=point_cloud_checkpoint_min_bytes,
    )
    if ensure_weights:
        ensure_checkpoint(MMDET_CHECKPOINT, MMDET_CHECKPOINT_URL, MMDET_MIN_BYTES)
        if point_cloud_spec.checkpoint_url:
            ensure_checkpoint(
                point_cloud_spec.checkpoint,
                point_cloud_spec.checkpoint_url,
                point_cloud_spec.min_bytes,
            )
        else:
            validate_local_checkpoint(point_cloud_spec.checkpoint, point_cloud_spec.min_bytes)
    thresholds = torch.tensor(
        list(depth_thresholds) if depth_thresholds is not None else [60.0, 60.0, 75.0],
        dtype=torch.float32,
    )
    if bev_width_meters <= 0.0 or bev_depth_meters <= 0.0 or bev_resolution <= 0.0:
        raise ValueError("BEV width/depth/resolution must be positive")
    bev_width = max(int(round(float(bev_width_meters) / float(bev_resolution))), 1)
    bev_height = max(int(round(float(bev_depth_meters) / float(bev_resolution))), 1)
    x_range = float(bev_width_meters) / 2.0
    z_range = float(bev_depth_meters) / 2.0
    image_adapter = MMDetInstanceSegAdapter(
        MMDET_CONFIG,
        MMDET_CHECKPOINT,
        device=device,
        score_thr=image_score_thr,
        max_dets=max_2d_dets,
        use_gt_boxes_in_train=not train_with_detector_2d,
    )
    point_cloud_adapter = MMDet3DNearDetectorAdapter(
        point_cloud_spec.config,
        point_cloud_spec.checkpoint,
        device=device,
        label_map=point_cloud_spec.label_map,
    )
    model = FarawayFrustumFramework(
        instance_2d_interface=image_adapter,
        near_3d_interface=point_cloud_adapter,
        depth_thresholds=thresholds,
        bev_height=bev_height,
        bev_width=bev_width,
        x_range=x_range,
        z_range=z_range,
        train_faraway_2d_iou_thr=match_2d_iou_thr,
        faraway_score_mode=faraway_score_mode,
        use_image_geometry_alignment=faraway_image_geometry_alignment,
        near_image_score_fusion=near_image_score_fusion,
        near_image_score_power=near_image_score_power,
        near_image_score_min_factor=near_image_score_min_factor,
        near_image_score_trainable=near_image_score_trainable,
        near_image_score_residual_scale=near_image_score_residual_scale,
    ).to(device)
    freeze_module(image_adapter)
    freeze_module(point_cloud_adapter)
    if train_near_score_only:
        freeze_module(model.faraway_refinement)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_class = torch.optim.AdamW if train_near_score_only else torch.optim.Adam
    optimizer = optimizer_class(trainable_params, lr=lr, weight_decay=weight_decay)
    system = FarawayFrustumSystem(
        model=model,
        optimizer=optimizer,
        criterion=FarawayFrustumCriterion(
            depth_thresholds=thresholds.tolist(),
            train_class_ids=train_class_ids,
            match_radius=criterion_match_radius,
            yaw_weight=yaw_loss_weight,
            score_weight=score_loss_weight,
            match_2d_iou_thr=match_2d_iou_thr,
            score_positive_radius=score_positive_radius,
            score_raw_miss_radius=score_raw_miss_radius,
            near_score_weight=near_score_loss_weight,
            near_preserve_weight=near_score_preserve_weight,
            near_rank_weight=near_rank_loss_weight,
            near_rank_margin=near_rank_margin,
            near_negative_only_weight=near_negative_only_weight,
            near_positive_iou_thr=near_positive_iou_thr,
            near_negative_iou_thr=near_negative_iou_thr,
            near_score_only=train_near_score_only,
        ),
        postprocessor=FarawayFrustumPostProcessor(
            KITTI_CLASS_NAMES,
            faraway_score_thr=faraway_score_thr,
            faraway_class_ids=faraway_infer_class_ids,
            faraway_near_suppress_radius=faraway_near_suppress_radius,
            paper_distance_fusion=paper_distance_fusion,
            pedestrian_far_distance=pedestrian_far_distance,
            car_far_distance=car_far_distance,
            faraway_nms_radius=faraway_nms_radius,
            faraway_topk_per_class=faraway_topk_per_class,
            car_score_thr=car_score_thr,
        ),
        image_adapter=image_adapter,
        point_cloud_adapter=point_cloud_adapter,
        device=device,
        class_names=KITTI_CLASS_NAMES,
    )
    verify_frozen_modules(system)
    return system


def train_step(system: FarawayFrustumSystem, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    system.image_adapter.eval()
    system.point_cloud_adapter.eval()
    system.model.train()
    system.image_adapter.eval()
    system.point_cloud_adapter.eval()
    batch = dict(batch)
    batch["mode"] = "train"
    system.optimizer.zero_grad(set_to_none=True)
    output = system.model.train_forward(batch)
    losses = system.criterion(output, batch)
    loss = losses["loss_total"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite Faraway loss: {loss.item()}")
    if not loss.requires_grad:
        detached = {key: value.detach() for key, value in losses.items()}
        detached["num_predictions"] = torch.tensor(output["boxes_3d_camera"].shape[0])
        detached["skipped_no_grad"] = torch.tensor(1)
        detached["grad_step"] = torch.tensor(0)
        return detached
    loss.backward()
    system.optimizer.step()
    detached = {key: value.detach() for key, value in losses.items()}
    detached["num_predictions"] = torch.tensor(output["boxes_3d_camera"].shape[0])
    detached["skipped_no_grad"] = torch.tensor(0)
    detached["grad_step"] = torch.tensor(1)
    return detached


@torch.no_grad()
def test_step(system: FarawayFrustumSystem, batch: Dict[str, Any], pred_dir: str) -> Dict[str, Any]:
    system.image_adapter.eval()
    system.point_cloud_adapter.eval()
    system.model.eval()
    batch = dict(batch)
    batch["mode"] = "test"
    output = system.model.test_forward(batch)
    path = system.postprocessor.write_kitti_prediction(output, batch, pred_dir)
    return {"output": output, "pred_path": path}


@torch.no_grad()
def raw_test_step(system: FarawayFrustumSystem, batch: Dict[str, Any], pred_dir: str) -> Dict[str, Any]:
    """只输出冻结 mmdet3d near detector 的 raw KITTI 结果，用于 raw/refined 对照。"""
    system.image_adapter.eval()
    system.point_cloud_adapter.eval()
    system.model.eval()
    batch = dict(batch)
    batch["mode"] = "test"
    raw = system.model.near_branch(system.point_cloud_adapter(batch))
    path = system.postprocessor.write_kitti_prediction(raw, batch, pred_dir)
    return {"output": raw, "pred_path": path}
