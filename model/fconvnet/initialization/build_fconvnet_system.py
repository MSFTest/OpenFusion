# -*- coding: utf-8 -*-
"""
build_fconvnet_system.py

作用：
    1) 用 mmdetection 构建 2D 检测器适配器
    2) 用 mmdet3d/KITTI batch 构建点云 Backbone 适配器
    3) 实例化 FConvNetFramework
    4) 提供 Criterion、PostProcessor、train_step 和 test_step
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.operator_registry  # noqa: F401
import utils.nms_bev  # noqa: F401
import utils.points_to_image  # noqa: F401
import utils.generate_depth_bins  # noqa: F401
import utils.compute_voxel_centroid  # noqa: F401
import utils.frustum_points_to_image  # noqa: F401
import utils.points_in_2d_boxes_mask  # noqa: F401
import utils.assign_values_to_bins  # noqa: F401
import utils.build_sliding_depth_bins_and_assign  # noqa: F401
import utils.group_points_by_indices  # noqa: F401
import utils.normalize_grouped_points_with_centroid  # noqa: F401
import utils.build_conv1d_feature_map  # noqa: F401
import common.masked_reduce_pool  # noqa: F401
import fusion_unit.symmetric.feature.ops.feature_concatenation  # noqa: F401
import fusion_unit.symmetric.feature.ops.align_and_concat_multiscale_1d  # noqa: F401

from model.fconvnet.fconvnet_framework import FConvNetFramework


MMDET_CONFIG_ROOT = Path("/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet/.mim/configs")
MMDET3D_CONFIG_ROOT = Path("/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet3d/.mim/configs")

DEFAULT_IMAGE_CFG = HIGH_ACCURACY_IMAGE_CFG = str(MMDET_CONFIG_ROOT / "dino/dino-5scale_swin-l_8xb2-36e_coco.py")
DEFAULT_IMAGE_CKPT = HIGH_ACCURACY_IMAGE_CKPT = "checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
DEFAULT_IMAGE_CKPT_URL = (
    "https://github.com/RistoranteRist/mmlab-weights/releases/download/"
    "dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
)

DEFAULT_POINT_CLOUD_CFG = str(MMDET3D_CONFIG_ROOT / "parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car.py")
DEFAULT_POINT_CLOUD_CKPT = "checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth"
DEFAULT_POINT_CLOUD_CKPT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/parta2/"
    "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car/"
    "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth"
)


def _freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _camera_boxes_iou3d_fconvnet(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Match KITTI length-on-x boxes to the shared rotated IoU helper's width-on-x convention."""
    from model.faraway_frustum.initialization.build_faraway_frustum_system import camera_boxes_iou3d

    order = [0, 1, 2, 4, 3, 5, 6]
    return camera_boxes_iou3d(boxes_a[:, order], boxes_b[:, order])


def _ensure_checkpoint(
    path: str | Path,
    url: str | None,
    required: bool = True,
    min_bytes: int = 1024,
) -> str:
    path = Path(path)
    if path.exists() and path.stat().st_size >= min_bytes:
        return str(path)
    if url is None:
        if required:
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[FConvNet] 下载权重: {url} -> {path}")
        urllib.request.urlretrieve(url, str(path))
    except Exception as exc:
        raise RuntimeError(
            f"无法下载权重，请手动下载到 {path}\n下载链接: {url}\n原始错误: {exc}"
        ) from exc
    if required and (not path.exists() or path.stat().st_size < min_bytes):
        raise RuntimeError(f"下载后的权重文件无效: {path}")
    return str(path)


class MMDetImageExtractor(nn.Module):
    """
    MMDetection 图像特征提取适配器。

    该类按任务要求保留，用于需要显式抽取图像特征的扩展场景。
    F-ConvNet 主流程默认使用 MMDet2DDetectorAdapter 产生 2D proposals。
    """

    def __init__(
        self,
        cfg_path: Optional[str],
        ckpt_path: Optional[str],
        device: str = "cuda:0",
        feat_index: Optional[int] = None,
    ):
        super().__init__()
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.feat_index = feat_index
        self.model = None
        if cfg_path:
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            from mmengine.registry import init_default_scope

            register_all_modules()
            init_default_scope("mmdet")
            self.model = init_detector(cfg_path, ckpt_path, device=device)
            self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        if self.model is None:
            return images
        feats = self.model.extract_feat(images)
        if self.feat_index is not None and isinstance(feats, (list, tuple)):
            return feats[self.feat_index]
        return feats


class MMDet2DDetectorAdapter(nn.Module):
    """
    MMDetection 2D 检测器适配器。

    输出字段：
        boxes_2d: (K, 4)
        scores_2d: (K,)
        labels_2d: (K,)
        batch_indices: (K,)
    """

    def __init__(
        self,
        cfg_path: Optional[str],
        ckpt_path: Optional[str],
        device: str = "cuda:0",
        score_thr: float = 0.05,
        max_per_img: int = 100,
        allowed_labels: Optional[tuple[int, ...]] = (2,),
        require_model: bool = True,
    ):
        super().__init__()
        self.device = device
        self.score_thr = float(score_thr)
        self.max_per_img = int(max_per_img)
        self.allowed_labels = tuple(int(x) for x in allowed_labels) if allowed_labels is not None else None
        self.model = None
        if cfg_path:
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
            from mmengine.registry import init_default_scope

            register_all_modules()
            init_default_scope("mmdet")
            self.model = init_detector(cfg_path, ckpt_path, device=device)
            self.model.eval()
        elif require_model:
            raise ValueError("MMDet2DDetectorAdapter requires image_cfg for real model inference")

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = batch["points"].device if torch.is_tensor(batch["points"]) else torch.device(self.device)
        if "det_boxes_2d" in batch:
            det_boxes = batch["det_boxes_2d"].to(device)
            det_scores = batch["det_scores_2d"].to(device)
            if det_boxes.dim() == 2:
                det_boxes, det_scores = det_boxes.unsqueeze(0), det_scores.unsqueeze(0)
            boxes_all, scores_all, batch_all = [], [], []
            for batch_idx, (boxes, scores) in enumerate(zip(det_boxes, det_scores)):
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes_all.append(boxes[valid])
                scores_all.append(scores[valid])
                batch_all.append(torch.full((int(valid.sum()),), batch_idx, device=device, dtype=torch.long))
            boxes, scores = torch.cat(boxes_all), torch.cat(scores_all)
            return {
                "boxes_2d": boxes,
                "scores_2d": scores,
                "labels_2d": torch.zeros((boxes.shape[0],), device=device, dtype=torch.long),
                "batch_indices": torch.cat(batch_all),
            }
        if bool(batch.get("use_gt_2d", False)):
            gt_boxes = batch["gt_boxes_2d"].to(device)
            if gt_boxes.dim() == 2:
                gt_boxes = gt_boxes.unsqueeze(0)
            boxes_all, batch_all, gt_indices_all = [], [], []
            for batch_idx, boxes in enumerate(gt_boxes):
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                gt_indices_all.append(torch.nonzero(valid, as_tuple=False).squeeze(1))
                boxes = boxes[valid]
                boxes_all.append(boxes)
                batch_all.append(
                    torch.full((boxes.shape[0],), batch_idx, device=device, dtype=torch.long)
                )
            boxes = torch.cat(boxes_all, dim=0)
            batch_indices = torch.cat(batch_all, dim=0)
            return {
                "boxes_2d": boxes,
                "scores_2d": torch.ones((boxes.shape[0],), device=device, dtype=boxes.dtype),
                "labels_2d": torch.zeros((boxes.shape[0],), device=device, dtype=torch.long),
                "batch_indices": batch_indices,
                "proposal_gt_indices": torch.cat(gt_indices_all, dim=0).long(),
            }
        if self.model is None:
            raise RuntimeError("MMDet2DDetectorAdapter has no real mmdet model")
        self.model.eval()

        from mmdet.apis import inference_detector
        from mmengine.registry import init_default_scope

        init_default_scope("mmdet")

        images = batch["images"].detach().cpu()
        img_list = []
        for img in images:
            arr = img.permute(1, 2, 0).numpy()
            arr = (arr * 255.0).clip(0, 255).astype("uint8")
            img_list.append(arr)
        results = inference_detector(self.model, img_list)
        if not isinstance(results, (list, tuple)):
            results = [results]
        boxes_all: List[torch.Tensor] = []
        scores_all: List[torch.Tensor] = []
        labels_all: List[torch.Tensor] = []
        batch_all: List[torch.Tensor] = []
        for b, res in enumerate(results):
            pred = res.pred_instances
            boxes = pred.bboxes.to(device)
            scores = pred.scores.to(device)
            labels = pred.labels.to(device)
            keep = scores >= self.score_thr
            if self.allowed_labels is not None:
                label_keep = torch.zeros_like(keep)
                for label_id in self.allowed_labels:
                    label_keep |= labels == label_id
                keep &= label_keep
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.shape[0] > self.max_per_img:
                order = scores.argsort(descending=True)[: self.max_per_img]
                boxes, scores, labels = boxes[order], scores[order], labels[order]
            labels = torch.zeros_like(labels)
            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels)
            batch_all.append(torch.full((boxes.shape[0],), b, device=device, dtype=torch.long))
        return {
            "boxes_2d": torch.cat(boxes_all, dim=0),
            "scores_2d": torch.cat(scores_all, dim=0),
            "labels_2d": torch.cat(labels_all, dim=0),
            "batch_indices": torch.cat(batch_all, dim=0),
        }


class FConvNetLiDARBackboneAdapter(nn.Module):
    """
    点云 Backbone 适配器。

    F-ConvNet 的点特征提取在网络内部完成，因此这里主要负责把 KITTI batch
    中的 raw points 与相机内参整理为 Framework 需要的输入。
    """

    def __init__(
        self,
        cfg_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: str = "cuda:0",
        enable_raw_detector: bool = False,
    ):
        super().__init__()
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.model = None
        self.enable_raw_detector = bool(enable_raw_detector)
        if cfg_path and self.enable_raw_detector:
            from mmdet3d.apis import init_model
            from mmdet3d.utils import register_all_modules
            from mmengine.registry import init_default_scope

            register_all_modules()
            init_default_scope("mmdet3d")
            self.model = init_model(cfg_path, ckpt_path, device=device)
            self.model.eval()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        points = batch["points"].to(self.device)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        cam_intrinsic = batch["cam_intrinsic"].to(self.device)
        out = {"points": points[..., :3], "cam_intrinsic": cam_intrinsic}
        pcd_paths = batch.get("pcd_path") or batch.get("velodyne_path")
        run_raw_detector = bool(batch.get("run_raw_detector", False))
        if self.model is not None and pcd_paths is not None and run_raw_detector:
            self.model.eval()
            try:
                from mmdet3d.apis import inference_detector
                from mmengine.registry import init_default_scope

                init_default_scope("mmdet3d")

                det3d_results = []
                for p in pcd_paths:
                    det3d_results.append(inference_detector(self.model, str(p)))
                out["det3d_results"] = det3d_results
            except Exception as exc:
                raise RuntimeError(f"真实 mmdet3d 模型推理失败: {exc}") from exc
        return out


class FConvNetCriterion(nn.Module):
    """Paper-style reference classification and component box regression losses."""

    def __init__(
        self,
        pos_radius: float = 2.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        num_heading_bins: int = 12,
        mean_size: tuple[float, float, float] = (1.6, 3.9, 1.56),
        inner_scale: float = 0.5,
        outer_scale: float = 1.0,
        refine_center_loss_weight: float = 1.0,
        refine_min_iou: float = 0.5,
    ):
        super().__init__()
        self.pos_radius = float(pos_radius)
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.num_heading_bins = int(num_heading_bins)
        self.inner_scale = float(inner_scale)
        self.outer_scale = float(outer_scale)
        if refine_center_loss_weight <= 0:
            raise ValueError("refine_center_loss_weight must be positive")
        self.refine_center_loss_weight = float(refine_center_loss_weight)
        if not 0.0 < refine_min_iou <= 1.0:
            raise ValueError("refine_min_iou must be in (0, 1]")
        self.refine_min_iou = float(refine_min_iou)
        self.register_buffer("mean_size", torch.tensor(mean_size, dtype=torch.float32))

    @staticmethod
    def _softmax_focal_loss_ignore(
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        keep = targets != -1
        probabilities = probabilities[keep]
        targets = targets[keep]
        if targets.numel() == 0:
            return probabilities.sum() * 0.0
        alpha_t = torch.where(
            targets == 0,
            probabilities.new_tensor(1.0 - alpha),
            probabilities.new_tensor(alpha),
        )
        probability_t = probabilities.gather(1, targets[:, None]).squeeze(1)
        loss = -alpha_t * (1.0 - probability_t).pow(gamma) * torch.log(probability_t + 1e-14)
        return loss.sum() / (targets > 0).sum().clamp_min(1)

    @staticmethod
    def _refinement_quality_mask(
        base_boxes: torch.Tensor,
        gt: torch.Tensor,
        proposal_batches: torch.Tensor,
        proposal_gt_indices: torch.Tensor,
    ) -> torch.Tensor:
        matched = gt[proposal_batches.long(), proposal_gt_indices.long()]
        gt_centers = matched[:, :3].clone()
        gt_centers[:, 1] -= matched[:, 5] / 2
        delta = base_boxes[:, :3] - gt_centers
        c, s = torch.cos(matched[:, 6]), torch.sin(matched[:, 6])
        local = torch.stack([
            c * delta[:, 0] - s * delta[:, 2],
            delta[:, 1],
            s * delta[:, 0] + c * delta[:, 2],
        ], dim=1).abs()
        center_limit = matched[:, [4, 5, 3]] * 0.5
        size_ratio = base_boxes[:, 3:6] / matched[:, 3:6].clamp_min(1e-3)
        yaw_error = torch.remainder(
            base_boxes[:, 6] - matched[:, 6] + torch.pi / 2, torch.pi
        ) - torch.pi / 2
        return (
            (local <= center_limit).all(dim=1)
            & (size_ratio >= 0.6).all(dim=1)
            & (size_ratio <= 1.4).all(dim=1)
            & (yaw_error.abs() <= torch.pi / 4)
        )

    @staticmethod
    def _rgb_refinement_matches(
        base_boxes: torch.Tensor,
        gt: torch.Tensor,
        proposal_batches: torch.Tensor,
        min_iou: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.full((base_boxes.shape[0],), -1, device=base_boxes.device, dtype=torch.long)
        quality = torch.zeros_like(indices, dtype=torch.bool)
        for batch_idx in proposal_batches.unique():
            proposal_idx = torch.nonzero(proposal_batches == batch_idx, as_tuple=False).squeeze(1)
            gt_row = gt[int(batch_idx.item())]
            valid_gt = torch.nonzero((gt_row[:, 3:6] > 0).all(dim=1), as_tuple=False).squeeze(1)
            if valid_gt.numel() == 0:
                continue
            boxes_bottom = base_boxes[proposal_idx].clone()
            boxes_bottom[:, 1] += boxes_bottom[:, 5] / 2
            ious = _camera_boxes_iou3d_fconvnet(boxes_bottom.detach(), gt_row[valid_gt].detach())
            best_iou, best = ious.max(dim=1)
            indices[proposal_idx] = valid_gt[best]
            quality[proposal_idx] = best_iou >= min_iou
        return indices, quality

    def _assign(
        self,
        refs: torch.Tensor,
        proposal_batches: torch.Tensor,
        gt: torch.Tensor,
        candidate_proposals: Optional[torch.Tensor] = None,
        proposal_gt_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = torch.zeros((refs.shape[0],), device=refs.device, dtype=torch.long)
        matched_boxes = refs.new_zeros((refs.shape[0], 7))
        matched_valid = torch.zeros((refs.shape[0],), device=refs.device, dtype=torch.bool)
        if candidate_proposals is not None and proposal_gt_indices is not None:
            for proposal_idx in candidate_proposals.unique():
                candidate_idx = torch.nonzero(
                    candidate_proposals == proposal_idx, as_tuple=False
                ).squeeze(1)
                batch_idx = int(proposal_batches[candidate_idx[0]].item())
                gt_idx = int(proposal_gt_indices[int(proposal_idx.item())].item())
                if gt_idx < 0 or gt_idx >= gt.shape[1]:
                    continue
                assigned = gt[batch_idx, gt_idx].unsqueeze(0).expand(candidate_idx.numel(), -1)
                if not bool((assigned[0, 3:6] > 0).all()):
                    continue
                assigned_centers = assigned[:, :3].clone()
                assigned_centers[:, 1] -= assigned[:, 5] / 2
                delta = refs[candidate_idx] - assigned_centers
                cos_yaw, sin_yaw = torch.cos(assigned[:, 6]), torch.sin(assigned[:, 6])
                local = torch.stack([
                    cos_yaw * delta[:, 0] - sin_yaw * delta[:, 2],
                    delta[:, 1],
                    sin_yaw * delta[:, 0] + cos_yaw * delta[:, 2],
                ], dim=1).abs()
                extents = assigned[:, [4, 5, 3]]
                inside_outer = (local <= extents * (self.outer_scale / 2)).all(dim=1)
                inside_inner = (local <= extents * (self.inner_scale / 2)).all(dim=1)
                labels[candidate_idx[inside_outer]] = -1
                labels[candidate_idx[inside_inner]] = 1
                nearest = candidate_idx[torch.linalg.vector_norm(delta, dim=1).argmin()]
                labels[nearest] = 1
                matched_boxes[candidate_idx] = assigned
                matched_valid[candidate_idx] = True
            return labels, matched_boxes, matched_valid
        for batch_idx in proposal_batches.unique():
            candidate_idx = torch.nonzero(proposal_batches == batch_idx, as_tuple=False).squeeze(1)
            boxes = gt[int(batch_idx.item())]
            boxes = boxes[(boxes[:, 3:6] > 0).all(dim=1)]
            if candidate_idx.numel() == 0 or boxes.numel() == 0:
                continue
            centers = boxes[:, :3].clone()
            centers[:, 1] -= boxes[:, 5] / 2
            distances = torch.cdist(refs[candidate_idx], centers)
            _, nearest = distances.min(dim=1)
            assigned = boxes[nearest]
            assigned_centers = centers[nearest]
            delta = refs[candidate_idx] - assigned_centers
            cos_yaw, sin_yaw = torch.cos(assigned[:, 6]), torch.sin(assigned[:, 6])
            local_x = cos_yaw * delta[:, 0] - sin_yaw * delta[:, 2]
            local_z = sin_yaw * delta[:, 0] + cos_yaw * delta[:, 2]
            local = torch.stack([local_x, delta[:, 1], local_z], dim=1).abs()
            extents = assigned[:, [4, 5, 3]]
            inside_outer = (local <= extents * (self.outer_scale / 2)).all(dim=1)
            inside_inner = (local <= extents * (self.inner_scale / 2)).all(dim=1)
            labels[candidate_idx[inside_outer]] = -1
            labels[candidate_idx[inside_inner]] = 1
            matched_boxes[candidate_idx] = assigned
            matched_valid[candidate_idx] = True
            for gt_idx in range(boxes.shape[0]):
                nearest_candidate = candidate_idx[distances[:, gt_idx].argmin()]
                labels[nearest_candidate] = 1
                matched_boxes[nearest_candidate] = boxes[gt_idx]
                matched_valid[nearest_candidate] = True
        return labels, matched_boxes, matched_valid

    @staticmethod
    def _camera_box_corners_from_centers(boxes: torch.Tensor) -> torch.Tensor:
        bottom_boxes = boxes.clone()
        bottom_boxes[:, 1] += bottom_boxes[:, 5] / 2
        return FConvNetPostProcessor._camera_box_corners(bottom_boxes)

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cls_logits = pred["cls_logits"].reshape(-1, 2)
        if cls_logits.shape[0] == 0:
            zero = torch.zeros((), device=cls_logits.device, dtype=cls_logits.dtype)
            return {
                "loss": zero, "cls_loss": zero, "reg_loss": zero,
                "num_pos": zero.long(), "num_neg": zero.long(), "num_ign": zero.long(),
            }
        refs_per_proposal = pred["center_refs"].shape[1]
        refs = pred["center_refs"].reshape(-1, 3)
        assignment_refs = pred.get("camera_center_refs", pred["center_refs"]).reshape(-1, 3)
        proposal_batches = pred.get("proposal_batch_indices")
        if proposal_batches is None:
            proposal_batches = torch.zeros(
                (pred["center_refs"].shape[0],), device=refs.device, dtype=torch.long
            )
        candidate_batches = proposal_batches.repeat_interleave(refs_per_proposal)
        gt = batch.get("gt_boxes_3d")
        if gt is None:
            gt = refs.new_zeros((int(candidate_batches.max().item()) + 1, 0, 7))
        gt = gt.to(device=refs.device, dtype=refs.dtype)
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        candidate_proposals = torch.arange(
            pred["center_refs"].shape[0], device=refs.device
        ).repeat_interleave(refs_per_proposal)
        proposal_gt_indices = pred.get("proposal_gt_indices")
        quality_mask = None
        if bool(pred.get("refine_residual", False)) and proposal_gt_indices is None:
            proposal_gt_indices, quality_mask = self._rgb_refinement_matches(
                pred["refine_base_boxes"], gt, proposal_batches, self.refine_min_iou
            )
        cls_target, matched_boxes, matched_valid = self._assign(
            assignment_refs,
            candidate_batches,
            gt,
            candidate_proposals=candidate_proposals,
            proposal_gt_indices=proposal_gt_indices,
        )
        candidate_valid = pred.get("candidate_valid_mask")
        if candidate_valid is not None:
            candidate_valid = candidate_valid.reshape(-1).bool()
            cls_target[~candidate_valid] = -1
            matched_valid[~candidate_valid] = False
        if bool(pred.get("refine_residual", False)) and proposal_gt_indices is not None:
            if quality_mask is None:
                quality_mask = self._refinement_quality_mask(
                    pred["refine_base_boxes"], gt, proposal_batches, proposal_gt_indices
                )
            candidate_quality = quality_mask.repeat_interleave(refs_per_proposal)
            cls_target[~candidate_quality] = -1
            matched_valid[~candidate_quality] = False
        cls_loss = self._softmax_focal_loss_ignore(
            F.softmax(cls_logits, dim=1), cls_target
        )
        pos = (cls_target == 1) & matched_valid
        zero = cls_logits.sum() * 0.0
        center_loss = heading_cls_loss = heading_res_loss = zero
        size_cls_loss = size_res_loss = corner_loss = zero
        if pos.any():
            raw = pred["reg_raw"].reshape(-1, pred["reg_raw"].shape[-1])[pos]
            refs_pos = refs[pos]
            target_boxes = matched_boxes[pos]
            target_centers = target_boxes[:, :3].clone()
            target_centers[:, 1] -= target_boxes[:, 5] / 2
            proposal_angles = pred.get("frustum_rot_angles")
            if proposal_angles is None:
                candidate_angles = refs.new_zeros((refs.shape[0],))
            else:
                candidate_angles = proposal_angles.repeat_interleave(refs_per_proposal).to(refs)
            pos_angles = candidate_angles[pos]
            base_centers = pred.get("refine_base_centers")
            if base_centers is not None:
                repeated_base = base_centers.repeat_interleave(refs_per_proposal, dim=0)
                target_centers -= repeated_base[pos].to(target_centers)
            cos_a, sin_a = torch.cos(pos_angles), torch.sin(pos_angles)
            center_x, center_z = target_centers[:, 0].clone(), target_centers[:, 2].clone()
            target_centers[:, 0] = cos_a * center_x - sin_a * center_z
            target_centers[:, 2] = sin_a * center_x + cos_a * center_z
            residual_refine = bool(pred.get("refine_residual", False))
            center_target = target_centers if residual_refine else target_centers - refs_pos
            center_loss = F.smooth_l1_loss(raw[:, :3], center_target)

            bins = self.num_heading_bins
            angle_per_bin = 2.0 * math.pi / bins
            target_heading = target_boxes[:, 6] - pos_angles
            shifted = torch.remainder(target_heading, 2.0 * math.pi)
            shifted = torch.remainder(shifted + angle_per_bin / 2, 2.0 * math.pi)
            heading_class = torch.floor(shifted / angle_per_bin).long()
            heading_target = (
                shifted - (heading_class.to(raw.dtype) * angle_per_bin + angle_per_bin / 2)
            ) / (angle_per_bin / 2)
            heading_logits = raw[:, 3:3 + bins]
            heading_residuals = raw[:, 3 + bins:3 + 2 * bins]
            heading_cls_loss = F.cross_entropy(heading_logits, heading_class)
            selected_heading_residual = heading_residuals.gather(1, heading_class[:, None]).squeeze(1)
            heading_res_loss = F.smooth_l1_loss(selected_heading_residual, heading_target)

            size_start = 3 + 2 * bins
            size_logits = raw[:, size_start:size_start + 1]
            size_cls_loss = size_logits.sum() * 0.0
            size_residual = raw[:, size_start + 1:size_start + 4]
            mean_size = self.mean_size.to(raw)
            if residual_refine:
                base_boxes = pred["refine_base_boxes"].repeat_interleave(
                    refs_per_proposal, dim=0
                )[pos].to(raw)
                size_base = base_boxes[:, 3:6]
            else:
                size_base = mean_size
            size_target = (target_boxes[:, 3:6] - size_base) / size_base
            size_res_loss = F.smooth_l1_loss(size_residual, size_target)

            predicted_center = raw[:, :3] if residual_refine else refs_pos + raw[:, :3]
            predicted_yaw = heading_class.to(raw.dtype) * angle_per_bin + selected_heading_residual * (angle_per_bin / 2)
            predicted_size = size_base * (1.0 + size_residual)
            predicted_boxes = torch.cat(
                [predicted_center, predicted_size, predicted_yaw[:, None]], dim=1
            )
            predicted_corners = self._camera_box_corners_from_centers(predicted_boxes)
            target_center_boxes = torch.cat(
                [target_centers, target_boxes[:, 3:6], target_heading[:, None]], dim=1
            )
            target_corners = self._camera_box_corners_from_centers(target_center_boxes)
            flipped = target_center_boxes.clone()
            flipped[:, 6] += math.pi
            flipped_corners = self._camera_box_corners_from_centers(flipped)
            corner_dist = torch.minimum(
                torch.linalg.vector_norm(predicted_corners - target_corners, dim=2).mean(dim=1),
                torch.linalg.vector_norm(predicted_corners - flipped_corners, dim=2).mean(dim=1),
            )
            corner_loss = F.smooth_l1_loss(corner_dist, torch.zeros_like(corner_dist))

        center_weight = self.refine_center_loss_weight if bool(pred.get("refine_residual", False)) else 1.0
        reg_loss = (
            center_weight * center_loss + heading_cls_loss + size_cls_loss
            + 20.0 * heading_res_loss + 20.0 * size_res_loss + 10.0 * corner_loss
        )
        loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "center_loss": center_loss,
            "heading_cls_loss": heading_cls_loss,
            "heading_res_loss": heading_res_loss,
            "size_res_loss": size_res_loss,
            "corner_loss": corner_loss,
            "num_pos": pos.sum(),
            "num_neg": (cls_target == 0).sum(),
            "num_ign": (cls_target == -1).sum(),
            "num_quality_proposals": (
                quality_mask.sum() if quality_mask is not None
                else torch.tensor(pred["center_refs"].shape[0], device=refs.device)
            ),
        }


class FConvNetPostProcessor:
    """
    F-ConvNet 后处理器。

    功能：
        - softmax 得分
        - score threshold
        - 旋转 3D NMS
        - KITTI label_2 格式文本输出
    """

    def __init__(self, score_thr: float = 0.05, nms_iou_thr: float = 0.1, max_per_sample: int = 100,
                 refine_score_weight: float = 0.0):
        if not 0.0 <= refine_score_weight <= 1.0:
            raise ValueError("refine_score_weight must be between 0 and 1")
        self.score_thr = float(score_thr)
        self.nms_iou_thr = float(nms_iou_thr)
        self.max_per_sample = int(max_per_sample)
        self.refine_score_weight = float(refine_score_weight)

    @staticmethod
    def _boxes_to_bev_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x, z, width, length = boxes[:, 0], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        return torch.stack(
            [x - width / 2, z - length / 2, x + width / 2, z + length / 2],
            dim=1,
        )

    @staticmethod
    def _nms_camera_boxes_3d(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        overlaps = _camera_boxes_iou3d_fconvnet(boxes, boxes)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            chosen = order[0]
            keep.append(chosen)
            order = order[1:]
            order = order[overlaps[chosen, order] <= threshold]
        return torch.stack(keep)

    @staticmethod
    def _camera_box_corners(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 8, 3))
        x, y, z, width, length, height, yaw = boxes.unbind(dim=1)
        x_local = torch.stack(
            [length / 2, length / 2, -length / 2, -length / 2] * 2, dim=1
        )
        y_local = torch.stack([torch.zeros_like(height)] * 4 + [-height] * 4, dim=1)
        z_local = torch.stack(
            [width / 2, -width / 2, -width / 2, width / 2] * 2, dim=1
        )
        cos_yaw, sin_yaw = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        corner_x = cos_yaw * x_local + sin_yaw * z_local + x[:, None]
        corner_y = y_local + y[:, None]
        corner_z = -sin_yaw * x_local + cos_yaw * z_local + z[:, None]
        return torch.stack([corner_x, corner_y, corner_z], dim=2)

    @classmethod
    def _project_camera_boxes(
        cls,
        boxes: torch.Tensor,
        p2: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        corners = cls._camera_box_corners(boxes)
        if corners.numel() == 0:
            return boxes.new_zeros((0, 4))
        ones = torch.ones((*corners.shape[:2], 1), device=boxes.device, dtype=boxes.dtype)
        projected = torch.cat([corners, ones], dim=2) @ p2.to(boxes).transpose(0, 1)
        uv = projected[..., :2] / projected[..., 2:3].clamp_min(1e-6)
        height, width = image_shape
        return torch.stack(
            [
                uv[..., 0].amin(dim=1).clamp(0, width - 1),
                uv[..., 1].amin(dim=1).clamp(0, height - 1),
                uv[..., 0].amax(dim=1).clamp(0, width - 1),
                uv[..., 1].amax(dim=1).clamp(0, height - 1),
            ],
            dim=1,
        )

    @staticmethod
    def _extract_mmdet3d_instances(det3d_result: Any):
        if isinstance(det3d_result, (tuple, list)) and len(det3d_result) > 0:
            det3d_result = det3d_result[0]
        if hasattr(det3d_result, "pred_instances_3d"):
            return det3d_result.pred_instances_3d
        if isinstance(det3d_result, dict):
            return det3d_result.get("pred_instances_3d")
        return None

    def _from_mmdet3d(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lidar_out = pred.get("lidar_out", {})
        det3d_results = lidar_out.get("det3d_results") if isinstance(lidar_out, dict) else None
        if not det3d_results:
            return None
        if "lidar2cam" not in batch or "P2" not in batch:
            return None

        from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes
        from mmdet3d.structures import points_cam2img

        sample_outputs = []
        for sample_idx, item in enumerate(det3d_results):
            inst = self._extract_mmdet3d_instances(item)
            if inst is None or not hasattr(inst, "bboxes_3d"):
                continue
            boxes_obj = inst.bboxes_3d
            boxes_lidar_obj = boxes_obj.clone() if hasattr(boxes_obj, "clone") else LiDARInstance3DBoxes(boxes_obj)
            scores = getattr(inst, "scores_3d", None)
            labels = getattr(inst, "labels_3d", None)
            if scores is None:
                continue
            lidar2cam = batch["lidar2cam"][sample_idx].detach().cpu().numpy()
            p2 = batch["P2"][sample_idx].to(device=boxes_lidar_obj.tensor.device, dtype=boxes_lidar_obj.tensor.dtype)
            boxes_cam_obj = boxes_lidar_obj.convert_to(Box3DMode.CAM, lidar2cam, correct_yaw=True)
            boxes = boxes_cam_obj.tensor.detach()
            boxes_lidar = boxes_lidar_obj.tensor.detach()
            corners_img = points_cam2img(boxes_cam_obj.corners, p2)
            minxy = torch.min(corners_img, dim=1)[0]
            maxxy = torch.max(corners_img, dim=1)[0]
            boxes_2d = torch.cat([minxy, maxxy], dim=1)

            H, W = batch.get("image_shape", (375, 1242))
            image_shape = boxes.new_tensor([float(W), float(H), float(W), float(H)])
            valid = (
                (boxes_2d[:, 0] < image_shape[0])
                & (boxes_2d[:, 1] < image_shape[1])
                & (boxes_2d[:, 2] > 0)
                & (boxes_2d[:, 3] > 0)
            )
            boxes_2d = torch.minimum(torch.maximum(boxes_2d, boxes_2d.new_zeros((4,))), image_shape)
            scores = scores.detach().to(device=boxes.device)
            labels = labels.detach().to(device=boxes.device) if labels is not None else torch.zeros_like(scores, dtype=torch.long)
            keep = scores >= self.score_thr
            keep &= valid
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            boxes_lidar = boxes_lidar[keep]
            boxes_2d = boxes_2d[keep]
            if boxes.shape[0] > self.max_per_sample:
                order = scores.argsort(descending=True)[: self.max_per_sample]
                boxes, scores, labels = boxes[order], scores[order], labels[order]
                boxes_lidar = boxes_lidar[order]
                boxes_2d = boxes_2d[order]
            alpha = -torch.atan2(-boxes_lidar[:, 1], boxes_lidar[:, 0]) + boxes[:, 6] if boxes.shape[0] else boxes.new_zeros((0,))
            sample_outputs.append({
                "boxes_3d": boxes[:, :7],
                "boxes_lidar": boxes_lidar[:, :7],
                "boxes_2d": boxes_2d,
                "alpha": alpha,
                "scores_3d": scores,
                "labels_3d": labels,
            })
        if not sample_outputs:
            return None
        first = sample_outputs[0]
        first["sample_results"] = sample_outputs
        first["source"] = "mmdet3d"
        return first

    def __call__(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, Any]:
        if pred.get("refine_residual", False) and "stage1_pred" in pred:
            stage1 = pred["stage1_pred"]
            stage1_foreground = F.softmax(stage1["cls_logits"], dim=-1)[..., 1]
            best = stage1_foreground.argmax(dim=1)
            rows = torch.arange(best.shape[0], device=best.device)
            foreground = F.softmax(pred["cls_logits"], dim=-1)[..., 1]
            valid = pred["candidate_valid_mask"].bool()
            best_ref = foreground.masked_fill(~valid, -1).argmax(dim=1)
            selected = pred["boxes_3d"][rows, best_ref]
            selected = torch.where(valid.any(dim=1)[:, None], selected, stage1["boxes_3d"][rows, best])
            merged = dict(stage1)
            merged["boxes_3d"] = stage1["boxes_3d"].clone()
            merged["boxes_3d"][rows, best] = selected
            if self.refine_score_weight:
                blended = (
                    (1.0 - self.refine_score_weight) * stage1_foreground[rows, best]
                    + self.refine_score_weight * foreground[rows, best_ref]
                ).clamp(1e-6, 1.0 - 1e-6)
                merged["cls_logits"] = stage1["cls_logits"].clone()
                merged["cls_logits"][rows, best, 0] = 0.0
                merged["cls_logits"][rows, best, 1] = torch.logit(blended)
            return self(merged, batch)
        logits = pred["cls_logits"]
        proposal_boxes_2d = pred.get("proposal_boxes_2d")
        proposal_scores = pred.get("proposal_scores")
        candidate_valid = pred.get("candidate_valid_mask")
        if pred.get("refine_base_centers") is not None and logits.dim() == 3:
            foreground_grid = F.softmax(logits, dim=-1)[..., 1]
            valid_grid = candidate_valid.bool() if candidate_valid is not None else torch.ones_like(
                foreground_grid, dtype=torch.bool
            )
            if bool(pred.get("refine_residual", False)):
                weights = foreground_grid * valid_grid.to(foreground_grid)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                box_grid = pred["boxes_3d"]
                linear = (box_grid[..., :6] * weights[..., None]).sum(dim=1)
                yaw = torch.atan2(
                    (torch.sin(box_grid[..., 6]) * weights).sum(dim=1),
                    (torch.cos(box_grid[..., 6]) * weights).sum(dim=1),
                )
                boxes = torch.cat([linear, yaw[:, None]], dim=1)
                selected_valid = valid_grid.any(dim=1)
                fallback_foreground = foreground_grid.max(dim=1).values
            else:
                best = foreground_grid.masked_fill(~valid_grid, -1).argmax(dim=1)
                rows = torch.arange(best.shape[0], device=best.device)
                boxes = pred["boxes_3d"][rows, best]
                selected_valid = valid_grid[rows, best]
                fallback_foreground = foreground_grid[rows, best]
            foreground_scores = pred.get(
                "base_foreground_scores", fallback_foreground
            ).to(foreground_grid)
            scores = foreground_scores
            if proposal_scores is not None:
                scores = scores + proposal_scores.to(scores)
            if proposal_boxes_2d is not None:
                proposal_boxes_2d = proposal_boxes_2d.to(boxes)
        else:
            boxes = pred["boxes_3d"].reshape(-1, 7)
            foreground_scores = F.softmax(logits.reshape(-1, 2), dim=-1)[:, 1]
            selected_valid = torch.ones_like(foreground_scores, dtype=torch.bool)
            scores = foreground_scores
            if proposal_scores is not None and logits.dim() == 3:
                scores = scores + proposal_scores.repeat_interleave(logits.shape[1]).to(scores)
            if proposal_boxes_2d is not None and logits.dim() == 3:
                proposal_boxes_2d = proposal_boxes_2d.repeat_interleave(
                    logits.shape[1], dim=0
                ).to(boxes)
        labels = torch.zeros((boxes.shape[0],), device=boxes.device, dtype=torch.long)
        boxes = torch.cat(
            [
                boxes[:, 0:1],
                boxes[:, 1:2] + boxes[:, 5:6] / 2,
                boxes[:, 2:],
            ],
            dim=1,
        )
        corners = self._camera_box_corners(boxes)
        keep = (foreground_scores >= self.score_thr) & selected_valid
        keep &= torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
        keep &= (boxes[:, 3:6] > 0).all(dim=1) & (boxes[:, 2] > 0.1)
        keep &= corners[..., 2].amin(dim=1) > 0.1
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if proposal_boxes_2d is not None:
            proposal_boxes_2d = proposal_boxes_2d[keep]
        if boxes.numel() > 0:
            keep_idx = self._nms_camera_boxes_3d(boxes, scores, self.nms_iou_thr)
            keep_idx = keep_idx[: self.max_per_sample]
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
            if proposal_boxes_2d is not None:
                proposal_boxes_2d = proposal_boxes_2d[keep_idx]
        result = {
            "boxes_3d": boxes,
            "scores_3d": scores,
            "labels_3d": labels,
            "source": "fconvnet",
        }
        if proposal_boxes_2d is not None:
            result["boxes_2d"] = proposal_boxes_2d
        elif "P2" in batch:
            p2 = batch["P2"][0] if batch["P2"].dim() == 3 else batch["P2"]
            result["boxes_2d"] = self._project_camera_boxes(
                boxes, p2, batch.get("image_shape", (375, 1242))
            )
            alpha = boxes[:, 6] - torch.atan2(boxes[:, 0], boxes[:, 2])
            result["alpha"] = torch.remainder(alpha + math.pi, 2.0 * math.pi) - math.pi
        return result

    def raw_detector(self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the diagnostic 3D-detector baseline without replacing F-ConvNet output."""
        return self._from_mmdet3d(pred, batch)

    def write_kitti(self, results: Dict[str, Any], pred_dir: str | Path, sample_id: str) -> Path:
        pred_dir = Path(pred_dir)
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_path = pred_dir / f"{sample_id}.txt"
        boxes = results["boxes_3d"].detach().cpu()
        scores = results["scores_3d"].detach().cpu()
        boxes_2d = results.get("boxes_2d")
        boxes_2d = boxes_2d.detach().cpu() if torch.is_tensor(boxes_2d) else None
        alpha = results.get("alpha")
        alpha = alpha.detach().cpu() if torch.is_tensor(alpha) else None
        with out_path.open("w", encoding="utf-8") as f:
            for idx, (box, score) in enumerate(zip(boxes, scores)):
                x, y, z, width, length, height, yaw = box[:7].tolist()
                if boxes_2d is not None and idx < boxes_2d.shape[0]:
                    x1, y1, x2, y2 = boxes_2d[idx].tolist()
                else:
                    x1, y1, x2, y2 = 0.0, 0.0, 50.0, 50.0
                alpha_val = float(alpha[idx]) if alpha is not None and idx < alpha.shape[0] else -10.0
                line = (
                    f"Car 0.00 0 {alpha_val:.4f} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} "
                    f"{height:.4f} {width:.4f} {length:.4f} {x:.4f} {y:.4f} {z:.4f} {yaw:.4f} {float(score):.6f}\n"
                )
                f.write(line)
        return out_path


@dataclass
class FConvNetSystem:
    model: FConvNetFramework
    optimizer: torch.optim.Optimizer
    criterion: FConvNetCriterion
    postprocessor: FConvNetPostProcessor
    device: str
    grad_clip_norm: Optional[float] = 10.0


def build_fconvnet_system(
    image_cfg: Optional[str] = None,
    image_ckpt: Optional[str] = None,
    point_cloud_cfg: Optional[str] = None,
    point_cloud_ckpt: Optional[str] = None,
    device: str = "cuda:0",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: Optional[float] = 10.0,
    **kwargs,
) -> FConvNetSystem:
    use_gt_2d = bool(kwargs.get("use_gt_2d", False))
    use_precomputed_2d = bool(kwargs.get("use_precomputed_2d", False))
    image_cfg = image_cfg or DEFAULT_IMAGE_CFG
    image_ckpt = image_ckpt or DEFAULT_IMAGE_CKPT
    point_cloud_cfg = point_cloud_cfg or DEFAULT_POINT_CLOUD_CFG
    point_cloud_ckpt = point_cloud_ckpt or DEFAULT_POINT_CLOUD_CKPT
    enable_raw_detector = bool(kwargs.get("enable_raw_detector", False))
    enable_refinement = bool(kwargs.get("enable_refinement", False))
    train_refinement_only = bool(kwargs.get("train_refinement_only", False))

    if not use_gt_2d and not use_precomputed_2d:
        image_ckpt = _ensure_checkpoint(
            image_ckpt,
            DEFAULT_IMAGE_CKPT_URL,
            required=True,
            min_bytes=kwargs.get("image_ckpt_min_bytes", 800 * 1024 * 1024),
        )
    if enable_raw_detector:
        point_cloud_ckpt = _ensure_checkpoint(
            point_cloud_ckpt,
            DEFAULT_POINT_CLOUD_CKPT_URL,
            required=True,
            min_bytes=kwargs.get("point_cloud_ckpt_min_bytes", 100 * 1024 * 1024),
        )

    detector2d = MMDet2DDetectorAdapter(
        None if (use_gt_2d or use_precomputed_2d) else image_cfg,
        None if (use_gt_2d or use_precomputed_2d) else image_ckpt,
        device=device,
        score_thr=kwargs.get("proposal_score_thr", 0.05),
        max_per_img=kwargs.get("max_2d_proposals", 100),
        allowed_labels=kwargs.get("allowed_2d_labels", (2,)),
        require_model=not (use_gt_2d or use_precomputed_2d),
    )
    lidar_backbone = FConvNetLiDARBackboneAdapter(
        point_cloud_cfg,
        point_cloud_ckpt,
        device=device,
        enable_raw_detector=enable_raw_detector,
    )
    _freeze_module(detector2d)
    _freeze_module(lidar_backbone)
    model = FConvNetFramework(
        detector2d_backbone=detector2d,
        point_cloud_backbone=lidar_backbone,
        bin_size=kwargs.get("bin_size", 0.25),
        stride=kwargs.get("stride", 0.25),
        max_points_per_frustum=kwargs.get("max_points_per_frustum", 128),
        max_depth=kwargs.get("max_depth", 70.0),
        enable_refinement=enable_refinement,
        train_refinement_only=train_refinement_only,
        refine_residual_scale=kwargs.get("refine_residual_scale", 1.0),
        refine_residual_gate_score=kwargs.get("refine_residual_gate_score", float("inf")),
    ).to(device)
    criterion = FConvNetCriterion(
        pos_radius=kwargs.get("pos_radius", 2.0),
        cls_weight=kwargs.get("cls_weight", 1.0),
        reg_weight=kwargs.get("reg_weight", 1.0),
        inner_scale=0.3 if enable_refinement else 0.5,
        outer_scale=0.6 if enable_refinement else 1.0,
        refine_center_loss_weight=kwargs.get("refine_center_loss_weight", 1.0),
        refine_min_iou=kwargs.get("refine_min_iou", 0.5),
    )
    postprocessor = FConvNetPostProcessor(
        score_thr=kwargs.get("score_thr", 0.05),
        nms_iou_thr=kwargs.get("nms_iou_thr", 0.1),
        max_per_sample=kwargs.get("max_per_sample", 100),
        refine_score_weight=kwargs.get("refine_score_weight", 0.0),
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("F-ConvNet has no trainable parameters")
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
    image_trainable = sum(parameter.numel() for parameter in detector2d.parameters() if parameter.requires_grad)
    point_cloud_trainable = sum(parameter.numel() for parameter in lidar_backbone.parameters() if parameter.requires_grad)
    optimizer_param_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(
        f"image_trainable={image_trainable} point_cloud_trainable={point_cloud_trainable} "
        f"optimizer_param_count={optimizer_param_count}"
    )
    return FConvNetSystem(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        postprocessor=postprocessor,
        device=device,
        grad_clip_norm=grad_clip_norm,
    )


def _move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) and k != "images" else v
    return out


def train_step(sys: FConvNetSystem, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    sys.model.train()
    sys.model.detector2d_backbone.eval()
    sys.model.point_cloud_backbone.eval()
    if bool(getattr(sys.model, "train_refinement_only", False)):
        sys.model.frustum_pointnet.eval()
        sys.model.frustum_fcn.eval()
        sys.model.detection_head.eval()
    batch = _move_batch_to_device(batch, sys.device)
    sys.optimizer.zero_grad(set_to_none=True)
    pred = sys.model.train_forward(batch)
    loss_dict = sys.criterion(pred, batch)
    loss = loss_dict["loss"]
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"non-finite F-ConvNet loss: {float(loss.detach().cpu())}")
    if not loss.requires_grad:
        result = {k: v.detach() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        result["skipped_no_grad"] = 1
        return result
    loss.backward()
    if sys.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(sys.model.parameters(), sys.grad_clip_norm)
    sys.optimizer.step()
    result = {k: v.detach() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    result["skipped_no_grad"] = 0
    return result


@torch.no_grad()
def test_step(
    sys: FConvNetSystem,
    batch: Dict[str, Any],
    include_raw: bool = False,
    refine_batch_stats: bool = False,
) -> Dict[str, Any]:
    sys.model.eval()
    if refine_batch_stats and getattr(sys.model, "refinement", None) is not None:
        for module in sys.model.refinement.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()
                module.momentum = 0.0
    batch = _move_batch_to_device(batch, sys.device)
    batch["run_raw_detector"] = include_raw
    pred = sys.model.test_forward(batch)
    result = sys.postprocessor(pred, batch)
    if include_raw:
        raw = sys.postprocessor(pred["stage1_pred"], batch) if "stage1_pred" in pred else sys.postprocessor.raw_detector(pred, batch)
        return {"fconvnet": result, "raw": raw}
    return result
