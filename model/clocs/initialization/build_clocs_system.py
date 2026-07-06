# -*- coding: utf-8 -*-
"""
build_clocs_system.py

作用：
    1) 用 mmdetection / mmdet3d 构建真实 2D/3D 检测器
    2) 实例化修改后的 CLOCSFramework
    3) 提供真实 train_step / test_step（默认只训练 FusionMLP）

说明：
    这版先把结构对齐、类别对齐、pair/post对象对齐修正到位。
    若后续要进一步逼近论文精度，建议继续把 2D/3D 候选改成 pre-NMS proposal。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# 激活自定义算子注册
import utils  # noqa: F401

from mmdet.apis import init_detector as init_det2d, inference_detector as infer_det2d
from mmdet3d.apis import init_model as init_det3d, inference_detector as infer_det3d

from backbone.det2d_interface import Det2DInterface, Det2DOutput, validate_det2d_output
from backbone.det3d_interface import Det3DInterface, Det3DOutput, validate_det3d_output
from model.clocs.clocs_framework import CLOCSFramework

from mmengine.registry import init_default_scope
from mmdet.utils import register_all_modules as register_mmdet_modules
from mmdet3d.utils import register_all_modules as register_mmdet3d_modules

register_mmdet_modules()
register_mmdet3d_modules()


def _normalize_allowed_labels(
    allowed_labels: Optional[Sequence[int]]
) -> Optional[set]:
    if allowed_labels is None:
        return None
    return {int(x) for x in allowed_labels}


# -----------------------------
# 2D Detector Adapter
# -----------------------------
class MMDet2DAdapter(Det2DInterface):
    """
    用 mmdetection 真实输出 2D boxes，并适配成：
        boxes_2d: (K,4)
        scores_2d: (K,)
        labels_2d: (K,)

    默认支持按类别过滤。
    对于 COCO 预训练 Faster R-CNN：
        car 类通常是 label=2
    """

    def __init__(self,
                 cfg_path: str,
                 ckpt_path: str,
                 device: str = "cuda:0",
                 score_thr: float = 0.05,
                 max_boxes: int = 200,
                 allowed_labels: Optional[Sequence[int]] = None,
                 debug: bool = True):
        super().__init__()
        self.device = device
        self.score_thr = float(score_thr)
        self.max_boxes = int(max_boxes)
        self.allowed_labels = _normalize_allowed_labels(allowed_labels)
        self.debug = bool(debug)

        init_default_scope("mmdet")
        self.model = init_det2d(cfg_path, ckpt_path, device=device)
        # 自动获取类别名称
        self.class_names = self.model.dataset_meta.get("classes", None)
        
        # 自动找到 car 对应的 label
        self.car_label = None
        if self.class_names is not None:
            for i, name in enumerate(self.class_names):
                if name.lower() == "car":
                    self.car_label = i
                    break
        
        # 覆盖 allowed_labels（保证正确）
        if self.car_label is not None:
            self.allowed_labels = {self.car_label}
    def _empty_output(self) -> Det2DOutput:
        out = Det2DOutput(
            boxes_2d=torch.zeros((0, 4), dtype=torch.float32, device=self.device),
            scores_2d=torch.zeros((0,), dtype=torch.float32, device=self.device),
            labels_2d=torch.zeros((0,), dtype=torch.long, device=self.device)
        )
        return validate_det2d_output(out)

    def _print_debug_stats(self, stage: str, scores: np.ndarray, labels: np.ndarray):
        if not self.debug:
            return

        num = int(scores.shape[0])
        if num == 0:
            return

        uniq, cnt = np.unique(labels, return_counts=True)
        stat = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)])

    def _post_filter(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Det2DOutput:
        if bboxes.size == 0:
            self._print_debug_stats("raw", np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            return self._empty_output()

        self._print_debug_stats("raw", scores, labels)

        keep = scores >= self.score_thr
        if self.allowed_labels is not None:
            keep = keep & np.isin(labels, list(self.allowed_labels))

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        # 单类 Car 任务：过滤后统一把 2D 标签映射成 0
        if labels.shape[0] > 0:
            labels = np.zeros_like(labels, dtype=np.int64)
        self._print_debug_stats("filtered", scores, labels)

        if bboxes.shape[0] == 0:
            return self._empty_output()

        order = np.argsort(-scores)
        if order.shape[0] > self.max_boxes:
            order = order[:self.max_boxes]

        bboxes = bboxes[order]
        scores = scores[order]
        labels = labels[order]

        self._print_debug_stats("topk", scores, labels)

        out = Det2DOutput(
            boxes_2d=torch.from_numpy(bboxes[:, :4]).to(self.device).float(),
            scores_2d=torch.from_numpy(scores).to(self.device).float(),
            labels_2d=torch.from_numpy(labels).to(self.device).long()
        )
        return validate_det2d_output(out)

    @torch.no_grad()
    def forward(self, images, image_meta: Optional[Dict[str, Any]] = None) -> Det2DOutput:
        init_default_scope("mmdet")
        self.model.eval()

        result = infer_det2d(self.model, images)

        # mmdet 3.x
        if hasattr(result, "pred_instances"):
            inst = result.pred_instances
            bboxes = torch.as_tensor(inst.bboxes).detach().cpu().numpy()
            scores = torch.as_tensor(inst.scores).detach().cpu().numpy()
            labels = torch.as_tensor(inst.labels).detach().cpu().numpy().astype(np.int64)
            return self._post_filter(bboxes, scores, labels)

        # mmdet 2.x
        all_boxes = []
        all_labels = []
        for cls_id, cls_boxes in enumerate(result):
            if cls_boxes is None or len(cls_boxes) == 0:
                continue
            cls_boxes = np.asarray(cls_boxes)
            if cls_boxes.ndim != 2 or cls_boxes.shape[1] < 5:
                continue
            all_boxes.append(cls_boxes[:, :5])
            all_labels.append(np.full((cls_boxes.shape[0],), cls_id, dtype=np.int64))

        if len(all_boxes) == 0:
            self._print_debug_stats("raw", np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            return self._empty_output()

        merged = np.concatenate(all_boxes, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)

        bboxes = merged[:, :4]
        scores = merged[:, 4]
        labels = merged_labels
        return self._post_filter(bboxes, scores, labels)

# -----------------------------
# 3D Detector Adapter
# -----------------------------
class MMDet3DAdapter(Det3DInterface):
    """
    用 mmdet3d 真实输出 3D boxes，并适配成：
        boxes_3d: (N,7) LiDAR [x,y,z,w,l,h,yaw]
        scores_3d: (N,)
        labels_3d: (N,)

    默认支持按类别过滤。
    KITTI Car 单类常见 label=0
    """

    def __init__(self,
                 cfg_path: str,
                 ckpt_path: str,
                 device: str = "cuda:0",
                 score_thr: float = 0.05,
                 max_boxes: int = 300,
                 allowed_labels: Optional[Sequence[int]] = None):
        super().__init__()
        self.device = device
        self.score_thr = float(score_thr)
        self.max_boxes = int(max_boxes)
        self.allowed_labels = _normalize_allowed_labels(allowed_labels)

        init_default_scope("mmdet3d")
        self.model = init_det3d(cfg_path, ckpt_path, device=device)

    @torch.no_grad()
    def forward(self, points, batch_meta: Optional[Dict[str, Any]] = None) -> Det3DOutput:
        init_default_scope("mmdet3d")
        self.model.eval()

        result, _data = infer_det3d(self.model, points)

        inst = result.pred_instances_3d
        b = inst.bboxes_3d
        s = inst.scores_3d if hasattr(inst, "scores_3d") else inst.scores
        l = inst.labels_3d if hasattr(inst, "labels_3d") else inst.labels

        boxes = torch.as_tensor(b.tensor, device=self.device, dtype=torch.float32)
        scores = torch.as_tensor(s, device=self.device, dtype=torch.float32).view(-1)
        labels = torch.as_tensor(l, device=self.device, dtype=torch.long).view(-1)

        if boxes.shape[1] > 7:
            boxes = boxes[:, :7]
        elif boxes.shape[1] < 7:
            raise ValueError(f"mmdet3d boxes dim < 7, got {boxes.shape}")

        keep = scores >= self.score_thr
        if self.allowed_labels is not None:
            allow = torch.zeros_like(labels, dtype=torch.bool)
            for cls_id in self.allowed_labels:
                allow |= (labels == int(cls_id))
            keep &= allow

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        # 单类 Car 任务：过滤后统一把 3D 标签映射成 0
        if labels.numel() > 0:
            labels = torch.zeros_like(labels)
        if scores.numel() > self.max_boxes:
            order = torch.argsort(scores, descending=True)[:self.max_boxes]
            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]

        out = Det3DOutput(boxes_3d=boxes, scores_3d=scores, labels_3d=labels)
        return validate_det3d_output(out, require_labels=True)


# -----------------------------
# Train/Test Runner
# -----------------------------
@dataclass
class CLOCSSystem:
    model: CLOCSFramework
    optimizer: torch.optim.Optimizer
    device: str = "cuda:0"
    pos_iou_thr: float = 0.1
    neg_iou_thr: float = 0.05


def _to_torch_calib(calib: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k in ["P2", "R0_rect", "Tr_velo_to_cam"]:
        v = calib[k]
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if not torch.is_tensor(v):
            raise TypeError(f"calib[{k}] must be np.ndarray or torch.Tensor")
        out[k] = v.to(device=device, dtype=torch.float32)
    return out


def build_clocs_system(
        det2d_cfg: str,
        det2d_ckpt: str,
        det3d_cfg: str,
        det3d_ckpt: str,
        device: str = "cuda:0",

        # 2D / 3D 候选过滤
        det2d_score_thr: float = 0.05,
        det2d_max_boxes: int = 200,
        det3d_score_thr: float = 0.05,
        det3d_max_boxes: int = 300,

        # 类别过滤（默认按 KITTI Car 任务）
        # COCO 的 car 一般是 2；KITTI 3D car 常见是 0
        det2d_allowed_labels: Optional[Sequence[int]] = (2,),
        det3d_allowed_labels: Optional[Sequence[int]] = (0,),

        # fusion 超参
        pair_iou_thr: float = 0.0,
        nms3d_thr: float = 0.5,
        score_fuse_alpha: float = 1.0,
        score_fuse_beta: float = 1.0,
        score_fuse_mode: str = "weighted_sum",

        # training
        lr: float = 1e-3,
        freeze_detectors: bool = True,
) -> CLOCSSystem:
    """
    返回一个真正可 train_step/test_step 的系统对象。
    """

    det2d = MMDet2DAdapter(
        det2d_cfg,
        det2d_ckpt,
        device=device,
        score_thr=det2d_score_thr,
        max_boxes=det2d_max_boxes,
        allowed_labels=det2d_allowed_labels,
        debug=False
    )

    det3d = MMDet3DAdapter(
        det3d_cfg,
        det3d_ckpt,
        device=device,
        score_thr=det3d_score_thr,
        max_boxes=det3d_max_boxes,
        allowed_labels=det3d_allowed_labels
    )

    model = CLOCSFramework(
        det2d_backbone=det2d,
        det3d_backbone=det3d,
        iou_threshold=pair_iou_thr,
        nms_threshold=nms3d_thr,
        score_fuse_alpha=score_fuse_alpha,
        score_fuse_beta=score_fuse_beta,
        score_fuse_mode=score_fuse_mode
    ).to(device)

    if freeze_detectors:
        for p in model.det2d.parameters():
            p.requires_grad_(False)
        for p in model.det3d.parameters():
            p.requires_grad_(False)

    train_params = [p for p in model.parameters() if p.requires_grad]
    if len(train_params) == 0:
        raise RuntimeError("No trainable parameters found. Check freeze_detectors or FusionMLP requires_grad.")

    optimizer = torch.optim.Adam(train_params, lr=lr)

    return CLOCSSystem(
        model=model,
        optimizer=optimizer,
        device=device
    )


# -----------------------------
# Real test function
# -----------------------------
def test_step(sys: CLOCSSystem, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    calib = _to_torch_calib(batch["calib"], sys.device)

    out = sys.model.test_forward({
        "images": batch["image"],
        "points": batch["points"],
        "calib": calib
    })

    return out


# -----------------------------
# Real train function
# -----------------------------
def train_step(sys: CLOCSSystem, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    真实 train：
        - 默认冻结 detector，只训练 FusionMLP
        - 监督信号：proposal 与 GT 的 IoU3D
        - 训练对象：每个 3D proposal 的 fused score（未做 NMS）
    """
    sys.model.train()
    sys.optimizer.zero_grad(set_to_none=True)

    calib = _to_torch_calib(batch["calib"], sys.device)

    # 1) detector 前向（默认冻结）
    with torch.no_grad():
        det2d_out = sys.model.det2d(batch["image"])
        det3d_out = sys.model.det3d(batch["points"])

    # 2) match + pair features
    pair_dict = sys.model.match_module(
        det2d_out.boxes_2d,
        det2d_out.scores_2d,
        getattr(det2d_out, "labels_2d", None),
        det3d_out.boxes_3d,
        det3d_out.scores_3d,
        getattr(det3d_out, "labels_3d", None),
        calib
    )

    # 3) FusionMLP 输出 pair_scores（这里有梯度）
    pair_scores = sys.model.score_module(pair_dict["pair_features"])
    if pair_scores.numel() == 0 or pair_dict["pair_indices"].numel() == 0:
        return {"loss_fusion": torch.tensor(0.0, device=sys.device)}

    # 4) 训练用 proposal 分数聚合（不做 NMS）
    score_matrix = torch.ops.fusionrepo.scatter_pairs_to_matrix(
        pair_scores,
        pair_dict["pair_indices"],
        pair_dict["K"],
        pair_dict["N"],
        0.0,
        True
    )
    
    fused_scores = torch.ops.fusionrepo.reduce_pool(
        score_matrix,
        dim=1,
        mode="max"
    ).view(-1)
    
    # 只保留“至少有一个 pair”的 3D 框
    has_pair = torch.zeros(
        pair_dict["N"],
        dtype=torch.bool,
        device=fused_scores.device
    )
    if pair_dict["pair_indices"].numel() > 0:
        has_pair[pair_dict["pair_indices"][:, 1].long()] = True

    # 5) GT
    gt_boxes = batch["gt_boxes_3d"].to(sys.device, dtype=torch.float32)
    if gt_boxes.numel() == 0 or pair_dict["boxes_3d"].numel() == 0:
        return {"loss_fusion": torch.tensor(0.0, device=sys.device)}

    # 单类时不强制要求 gt_labels_3d
    gt_labels = batch.get("gt_labels_3d", None)
    if gt_labels is not None:
        gt_labels = gt_labels.to(sys.device, dtype=torch.long).view(-1)

    prop_labels = pair_dict.get("labels_3d", None)
    if prop_labels is not None:
        prop_labels = prop_labels.to(sys.device, dtype=torch.long).view(-1)

    iou_mat = torch.ops.fusionrepo.iou3d_matrix(pair_dict["boxes_3d"], gt_boxes)  # (N,G)

    # 若有标签，则只允许同类 GT 提供监督
    if gt_labels is not None and prop_labels is not None and gt_labels.numel() > 0 and prop_labels.numel() > 0:
        same_class = (prop_labels.view(-1, 1) == gt_labels.view(1, -1))
        iou_mat = iou_mat * same_class.to(iou_mat.dtype)

    max_iou, _ = torch.max(iou_mat, dim=1)
    pos = max_iou >= sys.pos_iou_thr
    neg = max_iou <= sys.neg_iou_thr
    valid = (pos | neg) & has_pair


    if valid.sum().item() == 0:
        return {"loss_fusion": torch.tensor(0.0, device=sys.device)}

    targets = pos[valid].float()
    logits = fused_scores[valid]

    num_pos = targets.sum().item()
    num_neg = targets.numel() - num_pos

    if num_pos < 1:
        return {"loss_fusion": torch.tensor(0.0, device=sys.device)}

    pos_weight = torch.tensor(
        [num_neg / max(num_pos, 1.0)],
        device=logits.device,
        dtype=logits.dtype
    )
    logits = torch.clamp(logits, -10, 10)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight
    )


    loss.backward()
    sys.optimizer.step()



    return {"loss_fusion": loss.detach()}