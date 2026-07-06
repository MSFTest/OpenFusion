# -*- coding: utf-8 -*-
"""
AggregateAndPostModule

功能：
    将 pair 分数还原为 3D proposal 分数（fused_scores）
    并与原始 3D detector 分数（scores_3d）进行 re-ranking 融合
    最后进行 3D NMS

内部调用：
    - scatter_pairs_to_matrix
    - reduce_pool
    - nms3d
"""

import torch
import torch.nn as nn
from typing import Optional

class AggregateAndPostModule(nn.Module):
    def __init__(self,
                 nms_thresh: float = 0.5,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 fuse_mode: str = "weighted_sum"):
        """
        Args:
            nms_thresh: 3D NMS 阈值
            alpha: 原始 3D 分数 scores_3d 的权重
            beta: 融合增强分数 fused_scores 的权重
            fuse_mode: 融合方式
                - "weighted_sum": final = alpha*s3d + beta*sigmoid(sfuse)
                - "logit_add":   final = sigmoid(logit(s3d) + beta*sfuse)
                                 (要求 sfuse 为 logit 更合适；若你 sfuse 已经是概率，别用这个)
        """
        super().__init__()
        self.nms_thresh = nms_thresh
        self.alpha = alpha
        self.beta = beta
        self.fuse_mode = fuse_mode

    @staticmethod
    def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """把 (0,1) 概率转换为 logit，避免 0/1 数值问题。"""
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p / (1.0 - p))

    def forward(self,
                pair_scores: torch.Tensor,
                pair_indices: torch.Tensor,
                K: int,
                N: int,
                boxes_3d: torch.Tensor,
                scores_3d: torch.Tensor,
                labels_3d: Optional[torch.Tensor] = None):
        """
        Args:
            pair_scores: (p,) 每个 pair 的分数（通常来自 FusionMLP）
            pair_indices: (p,2) 每行是 [i_2d, j_3d]
            K: 2D 框数量
            N: 3D 框数量
            boxes_3d: (N,7)
            scores_3d: (N,)
            labels_3d: (N,) 可选，用于最终输出与 KITTI 评测

        Returns:
            final_boxes_3d: (M,7)
            final_scores_3d: (M,)
            keep_indices: (M,)
            final_labels_3d: (M,) 可选（若传入 labels_3d）
        """

        # 1️⃣ scatter 回 (1,K,N)
        score_matrix = torch.ops.fusionrepo.scatter_pairs_to_matrix(
            pair_scores,
            pair_indices,
            K,
            N
        )

        # 2️⃣ 沿 K 维 max 聚合 -> (1,N) 或 (N,)
        fused_scores = torch.ops.fusionrepo.reduce_pool(
            score_matrix,
            dim=1,
            mode="max"
        ).view(-1)  # (N,)

        # 3️⃣ 与原始 3D 分数进行 re-ranking 融合 -> final_scores (N,)
        s3d = scores_3d.to(device=fused_scores.device, dtype=fused_scores.dtype)

        if self.fuse_mode == "weighted_sum":
            # 把 fused_scores 当作 logit 更常见；这里用 sigmoid 转成 [0,1] 再加权更稳
            sfuse = torch.sigmoid(fused_scores)
            final_scores = self.alpha * s3d + self.beta * sfuse

        elif self.fuse_mode == "logit_add":
            # final = sigmoid(logit(s3d) + beta*fused_scores)
            # 更像“在原始分数上加一个 learned offset”
            final_scores = torch.sigmoid(self._safe_logit(s3d) + self.beta * fused_scores)

        else:
            raise ValueError("fuse_mode must be 'weighted_sum' or 'logit_add'")

        # 4️⃣ 3D NMS（使用 final_scores，而不是 fused_scores）
        keep_indices = torch.ops.fusionrepo.nms3d(
            boxes_3d,
            final_scores,
            self.nms_thresh
        )

        final_boxes_3d = boxes_3d[keep_indices]
        final_scores_3d = final_scores[keep_indices]

        if labels_3d is not None:
            final_labels_3d = labels_3d[keep_indices]
            return final_boxes_3d, final_scores_3d, keep_indices, final_labels_3d

        return final_boxes_3d, final_scores_3d, keep_indices