# -*- coding: utf-8 -*-
"""
CLOCSFramework

功能：
    融合框架调度层（不包含融合逻辑细节）

职责：
    1. 调用 2D / 3D backbone 接口生成 proposals
    2. 调用融合模块完成匹配、打分、聚合（包含 re-ranking + NMS）
    3. 提供 train / test 接口

注意：
    本文件不直接调用任何 fusionrepo 算子
    所有融合逻辑封装在 module 中
"""

import torch
import torch.nn as nn

from backbone.det2d_interface import Det2DInterface
from backbone.det3d_interface import Det3DInterface

from model.clocs.module.match_pair_module import MatchAndPairModule
from model.clocs.module.fusion_score_module import FusionScoreModule
from model.clocs.module.aggregate_post_module import AggregateAndPostModule


class CLOCSFramework(nn.Module):

    def __init__(self,
                 det2d_backbone: Det2DInterface,
                 det3d_backbone: Det3DInterface,
                 iou_threshold: float = 0.1,
                 nms_threshold: float = 0.5,
                 score_fuse_alpha: float = 1.0,
                 score_fuse_beta: float = 1.0,
                 score_fuse_mode: str = "weighted_sum"):
        super().__init__()

        self.det2d = det2d_backbone
        self.det3d = det3d_backbone

        self.match_module = MatchAndPairModule(iou_threshold=iou_threshold)
        self.score_module = FusionScoreModule()
        self.post_module = AggregateAndPostModule(
            nms_thresh=nms_threshold,
            alpha=score_fuse_alpha,
            beta=score_fuse_beta,
            fuse_mode=score_fuse_mode
        )

    def _run_fusion(self, images, points, calib):
        det2d_out = self.det2d(images)
        det3d_out = self.det3d(points)

        labels_2d = getattr(det2d_out, "labels_2d", None)
        labels_3d = getattr(det3d_out, "labels_3d", None)

        pair_dict = self.match_module(
            det2d_out.boxes_2d,
            det2d_out.scores_2d,
            labels_2d,
            det3d_out.boxes_3d,
            det3d_out.scores_3d,
            labels_3d,
            calib
        )

        pair_scores = self.score_module(pair_dict["pair_features"])
        
        # 后处理必须严格使用 pair_dict 里已经过滤后的 boxes_3d / scores_3d / labels_3d
        post_out = self.post_module(
            pair_scores,
            pair_dict["pair_indices"],
            pair_dict["K"],
            pair_dict["N"],
            pair_dict["boxes_3d"],
            pair_dict["scores_3d"],
            pair_dict.get("labels_3d", None)
        )
        
        if len(post_out) == 4:
            final_boxes_3d, final_scores_3d, keep_indices, final_labels_3d = post_out
            return {
                "boxes_3d": final_boxes_3d,
                "scores_3d": final_scores_3d,
                "labels_3d": final_labels_3d,
                "keep_indices": keep_indices,
                "pair_scores": pair_scores,
                "pair_dict": pair_dict,
                "det2d_out": det2d_out,
                "det3d_out": det3d_out
            }

        final_boxes_3d, final_scores_3d, keep_indices = post_out
        return {
            "boxes_3d": final_boxes_3d,
            "scores_3d": final_scores_3d,
            "keep_indices": keep_indices,
            "pair_scores": pair_scores,
            "pair_dict": pair_dict,
            "det2d_out": det2d_out,
            "det3d_out": det3d_out
        }

    # ===============================
    # Train Interface
    # ===============================
    def train_forward(self, batch: dict):
        """
        训练阶段接口
        """
        images = batch["images"]
        points = batch["points"]
        calib = batch["calib"]
        return self._run_fusion(images, points, calib)

    # ===============================
    # Test Interface
    # ===============================
    @torch.no_grad()
    def test_forward(self, batch: dict):
        """
        测试阶段接口
        """
        self.eval()
        images = batch["images"]
        points = batch["points"]
        calib = batch["calib"]
        return self._run_fusion(images, points, calib)