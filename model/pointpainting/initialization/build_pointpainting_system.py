# -*- coding: utf-8 -*-
"""
build_pointpainting_system.py

构建 PointPainting 训练/测试系统。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.point_cloud_feature_interface import PointCloudFeatureInterface
from model.pointpainting.pointpainting_framework import PointPaintingFramework


MMDET_CONFIG_ROOT = "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet/.mim/configs"
MMDET3D_CONFIG_ROOT = "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet3d/.mim/configs"

DEFAULT_IMAGE_CFG = (
    f"{MMDET_CONFIG_ROOT}/dino/"
    "dino-5scale_swin-l_8xb2-36e_coco.py"
)
DEFAULT_IMAGE_CKPT = (
    "checkpoints/"
    "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
)
DEFAULT_IMAGE_CKPT_URL = (
    "https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/"
    "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
)

DEFAULT_POINT_CLOUD_CFG = (
    f"{MMDET3D_CONFIG_ROOT}/pv_rcnn/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class.py"
)
DEFAULT_POINT_CLOUD_CKPT = (
    "checkpoints/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
)
DEFAULT_POINT_CLOUD_CKPT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
)


class MMDetImageExtractor(nn.Module):
    """
    MMDetection 图像特征提取适配器。

    输入:
        images: (B, 3, H, W)

    输出:
        feature: Tensor 或特征列表中的一层 Tensor。
    """

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str = "cuda:0",
        feat_index: int = 0,
    ):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet")
        self.device = device
        self.feat_index = int(feat_index)
        self.model = init_detector(cfg_path, ckpt_path, device=device)
        self.model.eval()

    def _build_data_samples(self, images: torch.Tensor):
        from mmdet.structures import DetDataSample

        h, w = images.shape[-2:]
        samples = []
        for _ in range(images.shape[0]):
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "ori_shape": (int(h), int(w)),
                    "img_shape": (int(h), int(w)),
                    "pad_shape": (int(h), int(w)),
                    "scale_factor": (1.0, 1.0),
                }
            )
            samples.append(sample)
        return samples

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        if not hasattr(self.model, "extract_feat"):
            raise AttributeError("mmdet model must provide extract_feat")
        feats = self.model.extract_feat(images)
        if isinstance(feats, (list, tuple)):
            return feats[self.feat_index]
        return feats

    @torch.no_grad()
    def semantic_score(self, images: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        用 MMDetection 分割/检测模型生成 PointPainting 语义分数图。

        Args:
            images: 图像张量，shape 为 (B, 3, H, W)。
            num_classes: PointPainting 语义通道数，默认对应
                [background, bicycle, vehicle, pedestrian, cyclist]。

        Returns:
            semantic_score_map，shape 为 (B, H, W, num_classes)。
        """
        h, w = images.shape[-2:]
        score_maps = images.new_full((images.shape[0], num_classes, h, w), 1e-4)
        score_maps[:, 0].fill_(1.0)

        if not hasattr(self.model, "data_preprocessor") or not hasattr(self.model, "predict"):
            raise AttributeError("mmdet model must provide data_preprocessor and predict")

        data = {
            "inputs": images,
            "data_samples": self._build_data_samples(images),
        }
        processed = self.model.data_preprocessor(data, training=False)
        preds = self.model.predict(processed["inputs"], processed["data_samples"], rescale=True)

        coco_to_pp = {
            0: 3,   # person -> pedestrian
            1: 1,   # bicycle -> bicycle
            2: 2,   # car -> vehicle
            5: 2,   # bus -> vehicle
            7: 2,   # truck -> vehicle
        }
        for batch_idx, pred in enumerate(preds):
            if hasattr(pred, "pred_instances") and "masks" in pred.pred_instances:
                inst = pred.pred_instances
                masks = inst.masks
                labels = inst.labels
                scores = getattr(inst, "scores", None)
                if hasattr(masks, "to_tensor"):
                    masks = masks.to_tensor(dtype=torch.bool, device=images.device)
                else:
                    masks = masks.to(device=images.device).bool()
                labels = labels.to(device=images.device)
                if scores is None:
                    scores = images.new_ones((masks.shape[0],))
                else:
                    scores = scores.to(device=images.device, dtype=images.dtype)
                for mask, label, score in zip(masks, labels, scores):
                    channel = coco_to_pp.get(int(label), None)
                    if channel is None or channel >= num_classes:
                        continue
                    if mask.shape[-2:] != (h, w):
                        mask = F.interpolate(
                            mask[None, None].float(), size=(h, w), mode="nearest"
                        )[0, 0].bool()
                    score_maps[batch_idx, 0, mask] = 1e-4
                    score_maps[batch_idx, channel, mask] = torch.maximum(
                        score_maps[batch_idx, channel, mask],
                        score.to(dtype=images.dtype),
                    )
            elif hasattr(pred, "pred_instances") and "bboxes" in pred.pred_instances:
                inst = pred.pred_instances
                bboxes = inst.bboxes.to(device=images.device, dtype=images.dtype)
                labels = inst.labels.to(device=images.device)
                scores = getattr(inst, "scores", None)
                if scores is None:
                    scores = images.new_ones((bboxes.shape[0],))
                else:
                    scores = scores.to(device=images.device, dtype=images.dtype)
                for bbox, label, score in zip(bboxes, labels, scores):
                    channel = coco_to_pp.get(int(label), None)
                    if channel is None or channel >= num_classes:
                        continue
                    x1, y1, x2, y2 = bbox.round().long()
                    x1 = int(x1.clamp(0, w - 1))
                    x2 = int(x2.clamp(0, w - 1))
                    y1 = int(y1.clamp(0, h - 1))
                    y2 = int(y2.clamp(0, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    score_maps[batch_idx, 0, y1 : y2 + 1, x1 : x2 + 1] = 1e-4
                    region = score_maps[batch_idx, channel, y1 : y2 + 1, x1 : x2 + 1]
                    score_maps[batch_idx, channel, y1 : y2 + 1, x1 : x2 + 1] = torch.maximum(
                        region,
                        score.to(dtype=images.dtype),
                    )

            if hasattr(pred, "pred_panoptic_seg"):
                panoptic = pred.pred_panoptic_seg.sem_seg.to(device=images.device)
                panoptic = panoptic[0] if panoptic.dim() == 3 else panoptic
                if panoptic.shape[-2:] != (h, w):
                    panoptic = F.interpolate(
                        panoptic[None, None].float(), size=(h, w), mode="nearest"
                    )[0, 0].long()
                metainfo = getattr(pred.pred_panoptic_seg, "metainfo", {})
                label_names = metainfo.get("label_names", None) if isinstance(metainfo, dict) else None
                for raw_id in torch.unique(panoptic):
                    raw_int = int(raw_id)
                    label = raw_int % 1000
                    channel = coco_to_pp.get(label, None)
                    if channel is None and label_names and label < len(label_names):
                        name = str(label_names[label]).lower()
                        if name in ("car", "truck", "bus"):
                            channel = 2
                        elif name == "person":
                            channel = 3
                        elif name == "bicycle":
                            channel = 1
                    if channel is None or channel >= num_classes:
                        continue
                    mask = panoptic == raw_id
                    score_maps[batch_idx, 0, mask] = 1e-4
                    score_maps[batch_idx, channel, mask] = 1.0

        return torch.softmax(score_maps, dim=1).permute(0, 2, 3, 1).contiguous()


def ensure_checkpoint(path: str, url: str) -> str:
    """
    检查权重文件，不存在则用 wget 下载。
    """
    ckpt_path = Path(path)
    if ckpt_path.exists() and ckpt_path.stat().st_size > 1024 * 1024:
        return str(ckpt_path)
    if ckpt_path.exists():
        ckpt_path.unlink()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-O", str(ckpt_path), url]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        raise RuntimeError(
            f"权重不存在且自动下载失败，请手动下载：\n{url}\n保存到：{ckpt_path}"
        ) from exc
    return str(ckpt_path)


class LightweightPointCloudBackbone(nn.Module):
    """
    轻量点云接口兜底实现，用于导入与 smoke 测试。
    """

    def __init__(self, in_channels: int = 9, hidden_channels: int = 64, num_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.score_head = nn.Linear(hidden_channels, num_classes)
        self.box_head = nn.Linear(hidden_channels, 7)

    def forward(self, points, **kwargs) -> dict:
        points_list = points if isinstance(points, (list, tuple)) else [points]
        rois = []
        scores = []
        labels = []
        flat_batch = []
        for b, pts in enumerate(points_list):
            if pts.numel() == 0:
                device = pts.device
                rois_b = torch.empty((0, 7), device=device, dtype=pts.dtype)
                scores_b = torch.empty((0,), device=device, dtype=pts.dtype)
                labels_b = torch.empty((0,), device=device, dtype=torch.long)
            else:
                feat = self.encoder(pts)
                cls = self.score_head(feat).softmax(dim=-1)
                score, label = cls.max(dim=-1)
                boxes = self.box_head(feat)
                xyz = pts[:, :3]
                boxes = torch.cat([xyz, boxes[:, 3:6].abs() + 0.5, boxes[:, 6:7]], dim=1)
                k = min(int(boxes.shape[0]), 32)
                top = torch.topk(score, k=k).indices
                rois_b = boxes[top]
                scores_b = score[top]
                labels_b = label[top]
            rois.append(rois_b)
            scores.append(scores_b)
            labels.append(labels_b)
            flat_batch.append(torch.full((rois_b.shape[0],), b, device=rois_b.device, dtype=torch.long))

        flat_rois = torch.cat(rois, dim=0) if rois else torch.empty((0, 7))
        flat_scores = torch.cat(scores, dim=0) if scores else torch.empty((0,))
        flat_labels = torch.cat(labels, dim=0) if labels else torch.empty((0,), dtype=torch.long)
        flat_batch_indices = torch.cat(flat_batch, dim=0) if flat_batch else torch.empty((0,), dtype=torch.long)
        return {
            "rois": rois,
            "roi_scores": scores,
            "roi_labels": labels,
            "flat_rois": flat_rois,
            "flat_roi_scores": flat_scores,
            "flat_roi_labels": flat_labels,
            "flat_roi_batch_indices": flat_batch_indices,
        }


def _bev_iou_axis_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    ax1 = boxes_a[:, 0] - boxes_a[:, 3] / 2
    ay1 = boxes_a[:, 1] - boxes_a[:, 4] / 2
    ax2 = boxes_a[:, 0] + boxes_a[:, 3] / 2
    ay2 = boxes_a[:, 1] + boxes_a[:, 4] / 2
    bx1 = boxes_b[:, 0] - boxes_b[:, 3] / 2
    by1 = boxes_b[:, 1] - boxes_b[:, 4] / 2
    bx2 = boxes_b[:, 0] + boxes_b[:, 3] / 2
    by2 = boxes_b[:, 1] + boxes_b[:, 4] / 2
    lt_x = torch.maximum(ax1[:, None], bx1[None])
    lt_y = torch.maximum(ay1[:, None], by1[None])
    rb_x = torch.minimum(ax2[:, None], bx2[None])
    rb_y = torch.minimum(ay2[:, None], by2[None])
    inter = (rb_x - lt_x).clamp(min=0) * (rb_y - lt_y).clamp(min=0)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-6)


class PointPaintingCriterion(nn.Module):
    """
    PointPainting 损失函数。

    使用 detector 输出框与 GT 的 BEV IoU 做简单匹配，计算 cls_loss 和 reg_loss。
    """

    def __init__(self, num_classes: int = 3, pos_iou_thr: float = 0.5, aux_weight: float = 0.02):
        super().__init__()
        self.num_classes = int(num_classes)
        self.pos_iou_thr = float(pos_iou_thr)
        self.aux_weight = float(aux_weight)

    def forward(self, pred: dict, batch: dict) -> dict:
        detector = pred["detector_output"]
        rois = detector.get("flat_rois", None)
        if rois is None or rois.numel() == 0 or "gt_boxes" not in batch:
            aux_loss = self._aux_loss(pred)
            return {"loss_cls": aux_loss * 0.0, "loss_reg": aux_loss * 0.0, "loss_aux": aux_loss}

        gt_boxes = batch["gt_boxes"]
        gt_labels = batch.get("gt_labels", torch.zeros((gt_boxes.shape[0],), device=rois.device, dtype=torch.long))
        if isinstance(gt_boxes, (list, tuple)):
            gt_boxes = gt_boxes[0]
        if isinstance(gt_labels, (list, tuple)):
            gt_labels = gt_labels[0]
        gt_boxes = gt_boxes.to(rois.device, rois.dtype)
        gt_labels = gt_labels.to(rois.device).long().clamp(min=0, max=self.num_classes - 1)

        iou = _bev_iou_axis_aligned(rois[:, :7], gt_boxes[:, :7])
        max_iou, matched = iou.max(dim=1) if iou.numel() else (
            torch.zeros((rois.shape[0],), device=rois.device),
            torch.zeros((rois.shape[0],), device=rois.device, dtype=torch.long),
        )
        pos = max_iou >= self.pos_iou_thr

        scores = detector.get("flat_roi_scores", torch.zeros((rois.shape[0],), device=rois.device, dtype=rois.dtype))
        loss_cls = F.binary_cross_entropy(scores.clamp(1e-4, 1 - 1e-4), pos.to(rois.dtype))
        if pos.any():
            loss_reg = F.smooth_l1_loss(rois[pos, :7], gt_boxes[matched[pos], :7])
        else:
            loss_reg = rois.sum() * 0.0
        loss_aux = self._aux_loss(pred)
        return {"loss_cls": loss_cls, "loss_reg": loss_reg, "loss_aux": loss_aux}

    def _aux_loss(self, pred: dict) -> torch.Tensor:
        aux_outputs = pred.get("aux_outputs", [])
        losses = []
        for out in aux_outputs:
            cls = out["cls_scores"]
            box = out["box_preds"]
            losses.append(cls.mean() * 0.0 + box.abs().mean() * self.aux_weight)
        if not losses:
            return torch.tensor(0.0)
        return sum(losses)


class PointPaintingPostProcessor:
    """
    PointPainting 后处理器。
    """

    def __init__(self, score_thr: float = 0.1, nms_thr: float = 0.1):
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)

    def __call__(self, pred: dict) -> dict:
        return pred


@dataclass
class PointPaintingSystem:
    model: PointPaintingFramework
    optimizer: torch.optim.Optimizer
    criterion: PointPaintingCriterion
    postprocessor: PointPaintingPostProcessor
    device: str
    grad_clip_norm: Optional[float] = 10.0


def _build_image_backbone(image_cfg, image_ckpt, device, feat_index):
    image_cfg = image_cfg or DEFAULT_IMAGE_CFG
    image_ckpt = image_ckpt or DEFAULT_IMAGE_CKPT
    if not Path(image_cfg).exists():
        raise FileNotFoundError(f"image config not found: {image_cfg}")
    image_ckpt = ensure_checkpoint(image_ckpt, DEFAULT_IMAGE_CKPT_URL)
    return MMDetImageExtractor(image_cfg, image_ckpt, device=device, feat_index=feat_index)


def _build_point_cloud_backbone(point_cloud_cfg, point_cloud_ckpt, device, painted_channels):
    if point_cloud_cfg == "lightweight":
        return LightweightPointCloudBackbone(in_channels=painted_channels).to(device)
    point_cloud_cfg = point_cloud_cfg or DEFAULT_POINT_CLOUD_CFG
    point_cloud_ckpt = point_cloud_ckpt or DEFAULT_POINT_CLOUD_CKPT
    if not Path(point_cloud_cfg).exists():
        raise FileNotFoundError(f"point cloud config not found: {point_cloud_cfg}")
    point_cloud_ckpt = ensure_checkpoint(point_cloud_ckpt, DEFAULT_POINT_CLOUD_CKPT_URL)
    from mmdet3d.apis import init_model
    from mmdet3d.utils import register_all_modules

    register_all_modules()
    detector = init_model(point_cloud_cfg, point_cloud_ckpt, device=device)
    return PointCloudFeatureInterface(detector).to(device)


def _freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _freeze_pretrained_backbones(model: PointPaintingFramework) -> None:
    """
    冻结已下载权重的 mmdet/mmdet3d 模型，只训练 PointPainting 自身模块。
    """
    _freeze_module(model.semantic_score_module.image_backbone)
    _freeze_module(model.painted_cloud_detector_module.point_cloud_backbone)


def _keep_frozen_backbones_eval(model: PointPaintingFramework) -> None:
    image_backbone = model.semantic_score_module.image_backbone
    point_cloud_backbone = model.painted_cloud_detector_module.point_cloud_backbone
    if image_backbone is not None:
        image_backbone.eval()
    if point_cloud_backbone is not None:
        point_cloud_backbone.eval()


def _trainable_parameters(model: nn.Module):
    return (param for param in model.parameters() if param.requires_grad)


def build_pointpainting_system(
    image_cfg: str | None = None,
    image_ckpt: str | None = None,
    point_cloud_cfg: str | None = None,
    point_cloud_ckpt: str | None = None,
    device: str = "cuda:0",
    num_semantic_classes: int = 5,
    num_classes: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_feat_index: int = 0,
    grad_clip_norm: Optional[float] = 10.0,
) -> PointPaintingSystem:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    image_backbone = _build_image_backbone(image_cfg, image_ckpt, device, image_feat_index)
    point_cloud_backbone = _build_point_cloud_backbone(
        point_cloud_cfg,
        point_cloud_ckpt,
        device,
        painted_channels=4 + int(num_semantic_classes),
    )
    model = PointPaintingFramework(
        image_backbone=image_backbone,
        point_cloud_backbone=point_cloud_backbone,
        num_semantic_classes=num_semantic_classes,
        num_classes=num_classes,
    ).to(device)
    _freeze_pretrained_backbones(model)
    criterion = PointPaintingCriterion(num_classes=num_classes).to(device)
    postprocessor = PointPaintingPostProcessor()
    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=float(lr), weight_decay=float(weight_decay))
    return PointPaintingSystem(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        postprocessor=postprocessor,
        device=device,
        grad_clip_norm=grad_clip_norm,
    )


def train_step(sys: PointPaintingSystem, batch: dict) -> dict:
    sys.model.train()
    _keep_frozen_backbones_eval(sys.model)
    sys.optimizer.zero_grad(set_to_none=True)
    pred = sys.model.train_forward(batch)
    losses = sys.criterion(pred, batch)
    total = sum(v for v in losses.values())
    total.backward()
    if sys.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(list(_trainable_parameters(sys.model)), float(sys.grad_clip_norm))
    sys.optimizer.step()
    losses["loss"] = total.detach()
    return {k: v.detach() if torch.is_tensor(v) else v for k, v in losses.items()}


@torch.no_grad()
def test_step(sys: PointPaintingSystem, batch: dict) -> dict:
    sys.model.eval()
    pred = sys.model.test_forward(batch)
    return sys.postprocessor(pred)
