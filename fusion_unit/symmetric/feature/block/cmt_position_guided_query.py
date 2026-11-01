# -*- coding: utf-8 -*-
"""CMT Position-Guided Query 生成与 point-based query denoising。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from fusion_unit.symmetric.feature.ops.feature_fusion_weighted_sum import (
    feature_fusion_weighted_sum,
)
from utils.cmt_anchor_multiview_projection import cmt_anchor_multiview_projection
from utils.cmt_projected_ray_coordinate_construction import cmt_projected_ray_coordinates


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """mmdetection 2.24 的 inverse_sigmoid 边界语义。"""
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class PositionGuidedQueryGenerator(nn.Module):
    """生成 CMT reference、DN query 与双模态位置引导 query。

    原源码位置：``cmt_head.py`` 的 ``reference_points``、``prepare_for_dn``、
    ``query_embed``。默认保存 900 个可学习归一化 LiDAR xyz anchors，初始化
    ``U(0,1)``。训练时按原 scalar/noise/split 规则构造 DN 前缀和 attention
    隔离 mask。BEV 与多视角 image 编码先各自执行，图像 mask 后沿视角求和，
    再通过公共逐元素加权求和原语按权重 1+1 融合，绝不平均。

    Args:
        bev_encoder: 编码 ``[B,N,2]`` 归一化 LiDAR-BEV 坐标的可训练模块。
        image_encoder: 编码 ``[B,V,N,D,3]`` 射线并消费 ``visible[B,V,N]`` 的模块。
        num_query: 匹配 query 数，正式配置为 900。
        pc_range: 米制 LiDAR 范围 ``[xmin,ymin,zmin,xmax,ymax,zmax]``。
        depth_num: 每条相机射线的深度点数，正式配置为 64。

    Returns:
        forward 返回 reference ``[B,N(+DN),3]``、融合位置 ``[B,N(+DN),C]``、
        bool attention mask 或 None、DN metadata 或 None；device 跟随 embedding。
    """

    def __init__(
        self,
        bev_encoder: nn.Module,
        image_encoder: nn.Module,
        num_query: int = 900,
        pc_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        depth_num: int = 64,
        scalar: int = 10,
        noise_scale: float = 1.0,
        noise_trans: float = 0.0,
        split: float = 0.75,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.bev_encoder = bev_encoder
        self.image_encoder = image_encoder
        self.reference_points = nn.Embedding(num_query, 3)
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        self.num_query = int(num_query)
        self.depth_num = int(depth_num)
        self.scalar = int(scalar)
        self.noise_scale = float(noise_scale)
        self.noise_trans = float(noise_trans)
        self.split = float(split)
        self.num_classes = int(num_classes)
        self.pc_range = tuple(float(value) for value in pc_range)

    def prepare_for_dn(
        self,
        batch_size: int,
        gt_boxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, object]]]:
        """按原 CMT 规则准备 DN reference、mask 与监督索引。"""
        reference = self.reference_points.weight
        if not self.training:
            return reference.unsqueeze(0).repeat(batch_size, 1, 1), None, None
        if (
            gt_boxes is None
            or gt_labels is None
            or len(gt_boxes) != batch_size
            or len(gt_labels) != batch_size
        ):
            raise ValueError("training requires one gt_boxes and gt_labels tensor per batch item")
        known_num = [int(item.size(0)) for item in gt_boxes]
        max_known = max(known_num)
        if max_known == 0:
            # 原实现会在 ``num_query // max(known_num)`` 处除零。空标注
            # 样本没有可重建目标，正确的 DN 语义是跳过 DN 前缀并继续使用
            # 正常 matching queries；非空输入路径保持逐句不变。
            return reference.unsqueeze(0).repeat(batch_size, 1, 1), None, None
        groups = min(self.scalar, self.num_query // max_known)
        labels = torch.cat(gt_labels).to(device=reference.device, dtype=torch.long)
        boxes = torch.cat(gt_boxes).to(device=reference.device)
        batch_idx = torch.cat(
            [
                torch.full(
                    (count,), index, dtype=torch.long, device=reference.device
                )
                for index, count in enumerate(known_num)
            ]
        )
        known_indice = torch.arange(labels.numel(), device=reference.device).repeat(groups)
        known_labels = labels.repeat(groups)
        known_labels_raw = labels.repeat(groups)
        known_bid = batch_idx.repeat(groups)
        known_bboxs = boxes.repeat(groups, 1)
        known_bbox_center = known_bboxs[:, :3].clone()
        known_bbox_scale = known_bboxs[:, 3:6].clone()
        if self.noise_scale > 0:
            diff = known_bbox_scale / 2 + self.noise_trans
            rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
            known_bbox_center += rand_prob * diff * self.noise_scale
            pc = known_bbox_center.new_tensor(self.pc_range)
            known_bbox_center = (known_bbox_center - pc[:3]) / (pc[3:] - pc[:3])
            known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
            known_labels[torch.norm(rand_prob, 2, 1) > self.split] = self.num_classes
        single_pad = max_known
        pad_size = single_pad * groups
        padded = torch.cat((reference.new_zeros((pad_size, 3)), reference), dim=0)
        padded = padded.unsqueeze(0).repeat(batch_size, 1, 1)
        base_indices = torch.cat(
            [torch.arange(count, device=reference.device) for count in known_num]
        )
        map_known_indice = torch.cat(
            [base_indices + single_pad * index for index in range(groups)]
        ).long()
        padded[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference.dtype)
        total = pad_size + self.num_query
        attn_mask = torch.zeros((total, total), device=reference.device, dtype=torch.bool)
        attn_mask[pad_size:, :pad_size] = True
        for index in range(groups):
            start, end = single_pad * index, single_pad * (index + 1)
            attn_mask[start:end, :start] = True
            attn_mask[start:end, end:pad_size] = True
        metadata: Dict[str, object] = {
            "known_indice": known_indice.long(),
            "batch_idx": batch_idx.long(),
            "map_known_indice": map_known_indice,
            "known_lbs_bboxes": (known_labels, known_bboxs),
            "known_labels_raw": known_labels_raw,
            "know_idx": [torch.ones_like(item, device=reference.device) for item in gt_labels],
            "pad_size": pad_size,
        }
        return padded, attn_mask, metadata

    def forward(
        self,
        batch_size: int,
        lidar2img: torch.Tensor,
        img2lidar: torch.Tensor,
        pad_h: int,
        pad_w: int,
        gt_boxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ):
        """投影 reference 并融合 BEV/image 位置编码。"""
        reference, attn_mask, metadata = self.prepare_for_dn(batch_size, gt_boxes, gt_labels)
        reference_for_pe = inverse_sigmoid(reference.clone()).sigmoid()
        pc_range = reference_for_pe.new_tensor(self.pc_range)
        bev_coords, projected_rays, visible = cmt_anchor_multiview_projection(
            reference_for_pe, lidar2img, pc_range, pad_h, pad_w
        )
        ray_coords = cmt_projected_ray_coordinates(
            projected_rays, img2lidar, pc_range, self.depth_num
        )
        bev_query = self.bev_encoder(bev_coords)
        image_query = self.image_encoder(ray_coords, visible)
        query_position = feature_fusion_weighted_sum(
            torch.stack((bev_query, image_query), dim=0), None
        )
        return reference, query_position, attn_mask, metadata

    @staticmethod
    def original_state_dict_key_map() -> dict:
        return {
            "reference_points.weight": "reference_points.weight",
            "bev_embedding.*": "bev_encoder.embedding.*",
            "rv_embedding.*": "image_encoder.embedding.*",
        }

    @staticmethod
    def remap_original_state_dict(original: dict) -> dict:
        """映射原 CmtHead reference 与共享坐标编码器参数。"""
        mapped = {}
        for key, value in original.items():
            if key.endswith("reference_points.weight"):
                mapped["reference_points.weight"] = value
            elif "bev_embedding." in key:
                mapped["bev_encoder.embedding." + key.split("bev_embedding.", 1)[1]] = value
            elif "rv_embedding." in key:
                mapped["image_encoder.embedding." + key.split("rv_embedding.", 1)[1]] = value
        return mapped
