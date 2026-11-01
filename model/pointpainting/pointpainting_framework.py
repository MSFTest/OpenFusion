# -*- coding: utf-8 -*-
"""
PointPaintingFramework

PointPainting 框架调度层。

注意：
    Framework 只调用 module 和 interface，不直接实现融合算子逻辑。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.pointpainting.module import (
    CyclistRefinementModule,
    KITTIResultModule,
    PaintedCloudDetectorModule,
    PaintedPointFeatureModule,
    PointProjectionModule,
    PointSemanticPaintingModule,
    SemanticScoreModule,
)


class PointPaintingFramework(nn.Module):
    """
    PointPainting 融合框架。

    Args:
        image_backbone:
            可选图像特征提取器，仅用于没有离线 semantic_score_map 时生成兜底分数图。

        point_cloud_backbone:
            点云特征提取/检测接口，接收 painted points。

        num_semantic_classes:
            语义分数通道数。

        num_classes:
            3D 检测类别数。
    """

    def __init__(
        self,
        image_backbone: nn.Module | None,
        point_cloud_backbone: nn.Module,
        num_semantic_classes: int = 5,
        num_classes: int = 3,
        box_dim: int = 7,
        cyclist_index: int = -1,
        cyclist_threshold: float = 0.3,
    ):
        super().__init__()
        self.semantic_score_module = SemanticScoreModule(
            image_backbone=image_backbone,
            num_semantic_classes=num_semantic_classes,
        )
        self.point_projection_module = PointProjectionModule()
        self.point_semantic_painting_module = PointSemanticPaintingModule()
        self.cyclist_refinement_module = CyclistRefinementModule(
            cyclist_index=cyclist_index,
            threshold=cyclist_threshold,
        )
        self.painted_point_feature_module = PaintedPointFeatureModule(
            in_channels=4 + int(num_semantic_classes),
            num_classes=num_classes,
            box_dim=box_dim,
        )
        self.painted_cloud_detector_module = PaintedCloudDetectorModule(point_cloud_backbone)
        self.kitti_result_module = KITTIResultModule()

    def _paint_single(
        self,
        points: torch.Tensor,
        semantic_score_map: torch.Tensor,
        calib: dict,
        image_shape: tuple[int, int] | None,
        batch_index: int,
    ) -> torch.Tensor:
        projection = self.point_projection_module(
            points,
            calib["Tr_velo2cam"],
            calib["P2"],
            calib["R0_rect"],
            image_shape,
        )
        painted = self.point_semantic_painting_module(
            points,
            projection["pixel_coords"],
            semantic_score_map,
            projection["valid_mask"],
            batch_index=batch_index,
        )
        return self.cyclist_refinement_module(painted)

    def _prepare_painted_points(self, batch: dict) -> tuple[list[torch.Tensor], list[dict]]:
        semantic_score_map = self.semantic_score_module(batch)
        points = batch["points"]
        points_list = list(points) if isinstance(points, (list, tuple)) else [points]
        calibs = batch.get("calibs", None)
        if calibs is None:
            calib = {
                "Tr_velo2cam": batch["Tr_velo2cam"],
                "P2": batch["P2"],
                "R0_rect": batch["R0_rect"],
            }
            calibs = [calib for _ in points_list]

        image_shape = batch.get("image_shape", None)
        if image_shape is None and "images" in batch:
            image_shape = tuple(batch["images"].shape[-2:])

        painted_points = []
        aux_outputs = []
        for b, pts in enumerate(points_list):
            painted = self._paint_single(
                pts,
                semantic_score_map,
                calibs[b],
                image_shape,
                b,
            )
            painted_points.append(painted)
            aux_outputs.append(self.painted_point_feature_module(painted))
        return painted_points, aux_outputs

    def train_forward(self, batch: dict) -> dict:
        painted_points, aux_outputs = self._prepare_painted_points(batch)
        detector_output = self.painted_cloud_detector_module(painted_points, batch)
        return {
            "painted_points": painted_points,
            "aux_outputs": aux_outputs,
            "detector_output": detector_output,
        }

    @torch.no_grad()
    def test_forward(self, batch: dict) -> dict:
        painted_points, aux_outputs = self._prepare_painted_points(batch)
        detector_output = self.painted_cloud_detector_module(painted_points, batch)
        kitti_lines = self.kitti_result_module(detector_output, batch)
        return {
            "painted_points": painted_points,
            "aux_outputs": aux_outputs,
            "detector_output": detector_output,
            "kitti_lines": kitti_lines,
        }

    def forward(self, batch: dict, mode: str = "test") -> dict:
        if mode == "train":
            return self.train_forward(batch)
        return self.test_forward(batch)
