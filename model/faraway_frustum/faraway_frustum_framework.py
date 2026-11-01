# -*- coding: utf-8 -*-
"""Faraway-Frustum framework 调度层。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from model.faraway_frustum.module import (
    FarawayMergeModule,
    FarawayRefinementModule,
    FrustumAngleModule,
    FrustumConstructionModule,
    KittiProjectionModule,
    NearImageScoreFusionModule,
    NearBranchModule,
)


def _boxes2d_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
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


class FarawayFrustumFramework(nn.Module):
    """Faraway-Frustum 调度框架。

    Framework 只调用 interface / adapter / module，不直接实现融合算子逻辑、
    投影细节或 KITTI 后处理。
    """

    def __init__(
        self,
        instance_2d_interface: nn.Module,
        near_3d_interface: nn.Module,
        depth_thresholds: torch.Tensor,
        projection_module: Optional[nn.Module] = None,
        bev_height: int = 300,
        bev_width: int = 200,
        x_range: float = 10.0,
        z_range: float = 15.0,
        train_faraway_2d_iou_thr: float = 0.3,
        faraway_score_mode: str = "image",
        use_image_geometry_alignment: bool = False,
        near_image_score_fusion: bool = False,
        near_image_score_power: float = 0.5,
        near_image_score_min_factor: float = 0.65,
        near_image_score_trainable: bool = False,
        near_image_score_residual_scale: float = 0.5,
    ):
        super().__init__()
        self.instance_2d_interface = instance_2d_interface
        self.near_3d_interface = near_3d_interface
        self.register_buffer("depth_thresholds", depth_thresholds.float())
        self.projection_module = projection_module if projection_module is not None else KittiProjectionModule()
        self.frustum_angle_module = FrustumAngleModule()
        self.train_faraway_2d_iou_thr = float(train_faraway_2d_iou_thr)
        if faraway_score_mode not in {"image", "product", "quality"}:
            raise ValueError("faraway_score_mode must be one of: image, product, quality")
        self.faraway_score_mode = faraway_score_mode
        self.use_image_geometry_alignment = bool(use_image_geometry_alignment)
        self.near_image_score_fusion = NearImageScoreFusionModule(
            enabled=near_image_score_fusion,
            power=near_image_score_power,
            min_factor=near_image_score_min_factor,
            trainable=near_image_score_trainable,
            residual_scale=near_image_score_residual_scale,
        )
        self.register_buffer(
            "class_size_priors",
            torch.tensor(
                [
                    [1.76, 0.66, 0.84],  # Pedestrian: h, w, l
                    [1.73, 0.60, 1.76],  # Cyclist: h, w, l
                    [1.53, 1.63, 3.88],  # Car: h, w, l
                ],
                dtype=torch.float32,
            ),
        )
        self.frustum_construction = FrustumConstructionModule()
        self.faraway_refinement = FarawayRefinementModule(
            bev_height=bev_height,
            bev_width=bev_width,
            x_range=x_range,
            z_range=z_range,
        )
        self.near_branch = NearBranchModule()
        self.merge_module = FarawayMergeModule()

    def _align_faraway_boxes_with_image_geometry(
        self,
        boxes_3d: torch.Tensor,
        source_boxes_2d: torch.Tensor,
        labels: torch.Tensor,
        p2: torch.Tensor,
    ) -> torch.Tensor:
        """用 2D 框高度和 P2 标定约束 Faraway camera 坐标框。

        远距离点云稀疏时，BEV 回归容易把中心深度推到错误的 frustum 点簇。
        这里使用 KITTI pinhole 几何：由 2D bbox 高度估计深度，再由 2D bbox
        中心/底边反投影 camera 坐标的 x、y。输出仍为
        ``(x, y, z, w, l, h, ry)``。
        """
        if boxes_3d.numel() == 0 or source_boxes_2d.numel() == 0:
            return boxes_3d
        p2 = p2.to(device=boxes_3d.device, dtype=boxes_3d.dtype)
        fx = p2[0, 0].clamp_min(1e-6)
        fy = p2[1, 1].clamp_min(1e-6)
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]
        x1, y1, x2, y2 = source_boxes_2d.unbind(dim=1)
        bbox_h = (y2 - y1).clamp_min(2.0)
        u_center = 0.5 * (x1 + x2)
        v_bottom = y2

        priors = self.class_size_priors.to(device=boxes_3d.device, dtype=boxes_3d.dtype)
        safe_labels = labels.clamp(min=0, max=priors.shape[0] - 1)
        class_hwl = priors[safe_labels]
        prior_h = class_hwl[:, 0]
        prior_w = class_hwl[:, 1]
        prior_l = class_hwl[:, 2]

        pred_h = boxes_3d[:, 5].clamp(1.2, 2.2)
        height_for_depth = 0.35 * pred_h + 0.65 * prior_h
        z_from_image = (fy * height_for_depth / bbox_h).clamp(3.0, 120.0)
        x_from_image = (u_center * z_from_image - cx * z_from_image - tx) / fx
        y_from_image = (v_bottom * z_from_image - cy * z_from_image - ty) / fy

        aligned = boxes_3d.clone()
        aligned[:, 0] = x_from_image
        aligned[:, 1] = y_from_image
        aligned[:, 2] = z_from_image
        aligned[:, 3] = 0.25 * boxes_3d[:, 3].clamp(0.8, 2.6) + 0.75 * prior_w
        aligned[:, 4] = 0.25 * boxes_3d[:, 4].clamp(2.0, 5.5) + 0.75 * prior_l
        aligned[:, 5] = height_for_depth
        return aligned

    def _run_modules(self, batch: Dict, train_mode: bool) -> Dict[str, torch.Tensor]:
        image_result = self.instance_2d_interface(batch)
        projection = self.projection_module(batch["points"], batch["calib"])
        boxes_2d = image_result["boxes_2d"]
        labels = image_result["labels"]
        masks = image_result["masks_2d"]
        scores_2d = image_result["scores"]

        frustum = self.frustum_construction(
            projection["points_2d_img"],
            projection["points_3d_cam0"],
            boxes_2d,
            masks,
            labels,
            self.depth_thresholds.to(device=boxes_2d.device),
        )

        is_faraway = frustum["is_faraway"]
        gt_boxes = batch.get("gt_boxes_3d_camera")
        gt_labels = batch.get("gt_labels")
        gt_boxes_2d = batch.get("gt_boxes_2d")
        if (
            train_mode
            and gt_boxes is not None
            and gt_labels is not None
            and gt_boxes_2d is not None
            and gt_boxes.numel() > 0
            and boxes_2d.numel() > 0
        ):
            gt_boxes_train = gt_boxes.to(device=boxes_2d.device, dtype=boxes_2d.dtype)
            gt_labels_train = gt_labels.to(device=boxes_2d.device, dtype=torch.long)
            gt_boxes_2d_train = gt_boxes_2d.to(device=boxes_2d.device, dtype=boxes_2d.dtype)
            ious = _boxes2d_iou(boxes_2d, gt_boxes_2d_train)
            best_iou, best_gt = ious.max(dim=1)
            matched_gt_labels = gt_labels_train[best_gt]
            gt_labels_safe = matched_gt_labels.clamp(
                min=0,
                max=max(int(self.depth_thresholds.numel()) - 1, 0),
            )
            gt_depth_thresholds = self.depth_thresholds.to(device=boxes_2d.device, dtype=boxes_2d.dtype)
            gt_faraway = gt_boxes_train[best_gt, 2] >= gt_depth_thresholds[gt_labels_safe]
            same_class = labels.to(device=boxes_2d.device, dtype=torch.long) == matched_gt_labels
            detector_gt_faraway = (
                (best_iou >= self.train_faraway_2d_iou_thr)
                & same_class
                & gt_faraway
                & (frustum["counts"] > 0)
            )
            is_faraway = is_faraway | detector_gt_faraway
        far_idx = torch.nonzero(is_faraway, as_tuple=False).squeeze(1)
        if far_idx.numel() > 0:
            frustum_angles = batch.get("frustum_angles")
            if frustum_angles is None or frustum_angles.numel() < labels.numel():
                frustum_angles = self.frustum_angle_module(
                    boxes_2d,
                    batch["calib"]["P2"].to(device=boxes_2d.device, dtype=boxes_2d.dtype),
                )
            faraway = self.faraway_refinement(
                frustum["clusters"][far_idx],
                frustum["counts"][far_idx],
                frustum["centroids"][far_idx],
                frustum_angles.to(device=boxes_2d.device, dtype=boxes_2d.dtype)[far_idx],
            )
            source_boxes = boxes_2d[far_idx]
            source_labels = labels[far_idx].to(device=boxes_2d.device, dtype=torch.long)
            if self.use_image_geometry_alignment:
                faraway["boxes_3d_camera"] = self._align_faraway_boxes_with_image_geometry(
                    faraway["boxes_3d_camera"],
                    source_boxes.to(dtype=faraway["boxes_3d_camera"].dtype),
                    source_labels,
                    batch["calib"]["P2"],
                )
            quality = torch.sigmoid(faraway["score_logits"]).to(dtype=faraway["boxes_3d_camera"].dtype)
            image_scores = scores_2d[far_idx].to(dtype=faraway["boxes_3d_camera"].dtype)
            if self.faraway_score_mode == "product":
                final_scores = image_scores * quality
            elif self.faraway_score_mode == "quality":
                final_scores = quality
            else:
                final_scores = image_scores
            faraway["scores"] = final_scores
            faraway["quality_scores"] = quality
            faraway["image_scores"] = image_scores
            faraway["labels"] = source_labels
            faraway["source_indices"] = far_idx
            faraway["source_boxes_2d"] = source_boxes
        else:
            faraway = {
                "boxes_3d_camera": boxes_2d.new_zeros((0, 7)),
                "scores": boxes_2d.new_zeros((0,)),
                "image_scores": boxes_2d.new_zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long, device=boxes_2d.device),
                "source_indices": torch.zeros((0,), dtype=torch.long, device=boxes_2d.device),
                "source_boxes_2d": boxes_2d.new_zeros((0, 4)),
                "regression_outputs": boxes_2d.new_zeros((0, 7)),
                "centroids": boxes_2d.new_zeros((0, 3)),
                "score_logits": boxes_2d.new_zeros((0,)),
                "bev_maps": boxes_2d.new_zeros((0, 1, 400, 700)),
            }

        detector_output = self.near_3d_interface(batch)
        near = self.near_branch(detector_output)
        near = self.near_image_score_fusion(
            near,
            image_result,
            batch["calib"]["P2"],
            batch["image_shape"],
        )
        merged = self.merge_module(faraway, near)
        merged.update(
            {
                "frustum": frustum,
                "faraway": faraway,
                "near": near,
                "image_instances": image_result,
                "projection": projection,
            }
        )
        return merged

    def train_forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return self._run_modules(batch, train_mode=True)

    def test_forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        return self._run_modules(batch, train_mode=False)

    def forward(self, batch: Dict, mode: str = "test") -> Dict[str, torch.Tensor]:
        if mode == "train":
            return self.train_forward(batch)
        if mode in ("test", "val", "predict"):
            return self.test_forward(batch)
        raise ValueError(f"Unsupported mode: {mode}")
