# -*- coding: utf-8 -*-
"""SparseFusion 双模态 proposal query 初始化模块。"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from common.bevdet_proposal_class_encoding import ProposalClassEncoding
from common.sparsefusion_layers import ConvLN, initialize_like_sparsefusion
from utils.bevdet_local_max_heatmap_nms import local_max_heatmap_nms
from utils.descending_argsort_topk import descending_argsort_topk
from utils.isfusion_gather_instance_features import isfusion_gather_instance_features


class _HeatmapHeadLN(nn.Sequential):
    def __init__(self, hidden: int, classes: int, layers: int = 2):
        modules = [ConvLN(hidden, hidden, 3, padding=1) for _ in range(layers - 1)]
        modules.append(nn.Conv2d(hidden, classes, 3, padding=1, bias=True))
        super().__init__(*modules)


class SparseFusionDualModalQueryInitializer(nn.Module):
    """从融合 BEV 和多尺度图像 heatmap 各选择 sparse query。

    原源码 ``sparsefusion_head_deform.py:364-413,762,797``。正式配置为
    10 类、LiDAR/image 各 200 proposals、六视图四尺度。heatmap 在排序前
    detach；局部 NMS 边缘为零，LiDAR nuScenes 类别8/9使用 kernel 1；排序
    使用 descending argsort。LiDAR/image 类别编码权重独立。
    """

    def __init__(self, hidden_channels=128, num_classes=10,
                 num_lidar_proposals=200, num_image_proposals=200,
                 views=6, levels=4, lidar_nms_kernel=3, image_nms_kernel=3):
        super().__init__()
        self.hidden_channels, self.num_classes = hidden_channels, num_classes
        self.num_lidar_proposals, self.num_image_proposals = num_lidar_proposals, num_image_proposals
        self.views, self.levels = views, levels
        self.lidar_nms_kernel, self.image_nms_kernel = lidar_nms_kernel, image_nms_kernel
        self.img_heatmap_head = nn.ModuleList([_HeatmapHeadLN(hidden_channels, num_classes) for _ in range(levels)])
        self.lidar_class_encoding = ProposalClassEncoding(hidden_channels, num_classes)
        self.image_class_encoding = ProposalClassEncoding(hidden_channels, num_classes)
        initialize_like_sparsefusion(self)

    def forward(self, lidar_bev: torch.Tensor, image_features: Sequence[torch.Tensor],
                cross_heatmap_logits: torch.Tensor, bev_pos: torch.Tensor,
                normal_image_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, channels, height, width = lidar_bev.shape
        lidar_heatmap = local_max_heatmap_nms(
            cross_heatmap_logits.detach().sigmoid(), self.lidar_nms_kernel, 0
        ).reshape(batch, self.num_classes, -1)
        lidar_top, _ = descending_argsort_topk(lidar_heatmap, self.num_lidar_proposals)
        lidar_class, lidar_index = lidar_top // (height * width), lidar_top % (height * width)
        lidar_query = isfusion_gather_instance_features(lidar_bev, lidar_index)
        lidar_query = self.lidar_class_encoding(lidar_query, lidar_class)
        lidar_position = bev_pos.gather(1, lidar_index[..., None].expand(-1, -1, 2))

        selected_levels, dense_levels, feature_levels = [], [], []
        for level, feature in enumerate(image_features):
            dense = self.img_heatmap_head[level](feature)
            selected = local_max_heatmap_nms(dense.detach().sigmoid(), self.image_nms_kernel, 1)
            size = feature.shape[-2] * feature.shape[-1]
            selected_levels.append(selected.reshape(batch, self.views, self.num_classes, size).permute(0, 2, 1, 3))
            dense_levels.append(dense.reshape(batch, self.views, self.num_classes, size).permute(0, 2, 1, 3))
            feature_levels.append(feature.reshape(batch, self.views, channels, size).permute(0, 2, 1, 3))
        heatmap_stack, dense_stack = torch.cat(selected_levels, -1), torch.cat(dense_levels, -1)
        positions_per_view = heatmap_stack.shape[-1]
        image_top, _ = descending_argsort_topk(heatmap_stack, self.num_image_proposals)
        image_class = image_top // (self.views * positions_per_view)
        image_view = image_top % (self.views * positions_per_view) // positions_per_view
        image_pos_index = image_top % positions_per_view
        image_index = image_top % (self.views * positions_per_view)
        image_stack = torch.cat(feature_levels, -1).reshape(batch, channels, self.views * positions_per_view)
        image_query = image_stack.gather(2, image_index[:, None].expand(-1, channels, -1))
        image_query = self.image_class_encoding(image_query, image_class)
        image_position = normal_image_positions.expand(batch, -1, -1).gather(
            1, image_pos_index[..., None].expand(-1, -1, 2)
        )
        return {
            "lidar_query_feat": lidar_query, "lidar_query_pos": lidar_position,
            "lidar_query_class": lidar_class, "lidar_query_index": lidar_index,
            "lidar_heatmap": lidar_heatmap,
            "lidar_query_heatmap_score": lidar_heatmap.gather(2, lidar_index[:, None].expand(-1, self.num_classes, -1)),
            "image_query_feat": image_query, "image_query_pos": image_position,
            "image_query_view": image_view, "image_query_class": image_class,
            "image_query_index": image_index, "image_query_pos_index": image_pos_index,
            "image_heatmap": heatmap_stack.reshape(batch, self.num_classes, -1),
            "image_dense_heatmap": dense_stack,
            "image_query_heatmap_score": heatmap_stack.reshape(batch, self.num_classes, -1).gather(
                2, image_index[:, None].expand(-1, self.num_classes, -1)),
        }

    @staticmethod
    def original_state_dict_key_map():
        return {
            "class_encoding.": "lidar_class_encoding.class_encoding.",
            "img_class_encoding.": "image_class_encoding.class_encoding.",
            "img_heatmap_head.": "img_heatmap_head.",
        }
