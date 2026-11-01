# -*- coding: utf-8 -*-
"""DeepInteraction 融合感知 query 初始化块。"""

import copy

import torch
from torch import nn

from common.bevdet_proposal_class_encoding import ProposalClassEncoding
from utils.bevdet_compute_query_position import compute_query_position
from utils.bevdet_local_max_heatmap_nms import local_max_heatmap_nms
from utils.isfusion_gather_instance_features import isfusion_gather_instance_features


class FusionAwareQueryInitialization(nn.Module):
    """融合两个 dense heatmap，并从融合前 LiDAR BEV 初始化 query。

    Args:
        channels: BEV/query 通道，正式配置 128。
        num_classes: nuScenes 类别数，正式配置 10。
        nms_kernel: 局部 NMS kernel，正式配置 3。
    Returns:
        输入 ``pre_fusion_lidar/fused_lidar [B,C,H,W]`` 与 proposal 数，返回
        query/state 字典。跨类别选择严格使用稳定语义的完整降序 argsort，
        不以 topk 替换；类别 8/9 使用 nuScenes kernel=1 例外。

    heatmap 融合、NMS、类别编码、位置计算和空间 gather 分别复用现有真实
    算子/层。``pre_fusion_flat`` 保留给初始 Transformer；``fused_flat``
    保留给后续 LiDAR MMPI。原源码：``deepinteraction_decoder.py:223-254``。
    """
    def __init__(self, channels=128, num_classes=10, nms_kernel=3):
        super().__init__()
        self.num_classes = int(num_classes)
        self.nms_kernel = int(nms_kernel)
        heatmap_head = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            ),
            nn.Conv2d(channels, num_classes, 3, padding=1, bias=True),
        )
        self.heatmap_lidar = heatmap_head
        self.heatmap_fused = copy.deepcopy(heatmap_head)
        self.class_encoding = ProposalClassEncoding(channels, num_classes)

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 heatmap ``ConvModule`` 与直接 class_encoding checkpoint 键。"""
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("heatmap_head.0.conv."):
                key = "heatmap_lidar.0.0." + key[len("heatmap_head.0.conv."):]
            elif key.startswith("heatmap_head.0.bn."):
                key = "heatmap_lidar.0.1." + key[len("heatmap_head.0.bn."):]
            elif key.startswith("heatmap_head.1."):
                key = "heatmap_lidar.1." + key[len("heatmap_head.1."):]
            elif key.startswith("heatmap_head_img.0.conv."):
                key = "heatmap_fused.0.0." + key[len("heatmap_head_img.0.conv."):]
            elif key.startswith("heatmap_head_img.0.bn."):
                key = "heatmap_fused.0.1." + key[len("heatmap_head_img.0.bn."):]
            elif key.startswith("heatmap_head_img.1."):
                key = "heatmap_fused.1." + key[len("heatmap_head_img.1."):]
            elif key.startswith("class_encoding."):
                key = key.replace(
                    "class_encoding.", "class_encoding.class_encoding.", 1
                )
            mapped[key] = value
        return mapped

    def forward(self, pre_fusion_lidar, fused_lidar, num_proposals):
        batch, channels, height, width = pre_fusion_lidar.shape
        old_flat = pre_fusion_lidar.flatten(2)
        fused_flat = fused_lidar.flatten(2)
        old_dense = self.heatmap_lidar(pre_fusion_lidar)
        fused_dense = self.heatmap_fused(fused_lidar)
        heatmap = (old_dense.detach().sigmoid() + fused_dense.detach().sigmoid()) / 2
        heatmap = local_max_heatmap_nms(heatmap, self.nms_kernel, 0)
        flat_heatmap = heatmap.flatten(2)
        top = flat_heatmap.reshape(batch, -1).argsort(
            dim=-1, descending=True
        )[..., :int(num_proposals)]
        classes = torch.div(top, height * width, rounding_mode="floor")
        indices = torch.remainder(top, height * width)
        # 二次覆盖检查：现有 IS-Fusion gather 已完全覆盖，禁止新增同功能 schema。
        query = isfusion_gather_instance_features(pre_fusion_lidar, indices)
        query = self.class_encoding(query, classes)
        bev_position = compute_query_position(
            torch.arange(height * width, device=query.device).reshape(1, -1)
            .expand(batch, -1), height, width
        ).to(query.dtype)
        query_position = compute_query_position(indices, height, width).to(query.dtype)
        query_score = flat_heatmap.gather(
            2, indices[:, None].expand(-1, self.num_classes, -1)
        )
        return {
            "query": query,
            "query_position": query_position,
            "pre_fusion_flat": old_flat,
            "fused_flat": fused_flat,
            "bev_position": bev_position,
            "query_heatmap_score": query_score,
            "dense_heatmap": fused_dense,
            "query_labels": classes,
        }
