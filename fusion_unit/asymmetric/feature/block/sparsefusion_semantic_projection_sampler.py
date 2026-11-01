# -*- coding: utf-8 -*-
"""SparseFusion semantic transfer 的学习投影与多尺度采样 block。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch
from torch import nn

from common.sparsefusion_layers import SparseFusionLinearLayerNorm, initialize_like_sparsefusion
from utils.sample_features_by_grid_v2 import sample_features_by_grid_v2


class SparseFusionSemanticProjectionSampler(nn.Module):
    """投影 LiDAR/image 特征，并在 BEV 投影位置采样四尺度图像语义。

    原源码 ``sparsefusion_head_deform.py:607-702``。每个可见 BEV query 从
    同一相机各尺度以 bilinear/border/align_corners=False 采样，跨尺度逐元素
    max 后与投影 LiDAR query 相加。返回按样本最大可见数补齐的 ragged 容器；
    索引离散不可微，采样与投影可微。
    """

    def __init__(self, hidden_channels=128, views=6, levels=4):
        super().__init__()
        self.hidden_channels, self.views, self.levels = hidden_channels, views, levels
        self.heatmap_pts_proj = SparseFusionLinearLayerNorm(hidden_channels, hidden_channels)
        self.heatmap_img_proj = SparseFusionLinearLayerNorm(hidden_channels, hidden_channels)
        initialize_like_sparsefusion(self)

    def forward(self, lidar_bev, image_features: Sequence[torch.Tensor], reference_points,
                query_positions, visible_mask, normal_image_positions, spatial_shapes,
                level_start_index, input_padding_mask) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        batch, channels, height, width = lidar_bev.shape
        projected_images, flattened_images = [], []
        for feature in image_features:
            projected = self.heatmap_img_proj(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            projected = projected.reshape(batch, self.views, channels, *projected.shape[-2:])
            projected_images.append(projected)
            flattened_images.append(projected.flatten(-2))
        image_stack = torch.cat(flattened_images, -1)
        projected_lidar = self.heatmap_pts_proj(lidar_bev.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        lidar_flat = projected_lidar.flatten(2)
        query_features, positions_out, refs_out, indices_out, padding_out = [], [], [], [], []
        for sample in range(batch):
            counts, max_visible = visible_mask[sample].sum(-1), int(visible_mask[sample].sum(-1).max().item())
            query_feature = torch.zeros((self.views, channels, max_visible), device=lidar_bev.device)
            query_position = torch.zeros((self.views, max_visible, 3), device=lidar_bev.device)
            refs = torch.zeros((self.views, max_visible, self.levels, 2), device=lidar_bev.device)
            indices = torch.zeros((self.views, max_visible), dtype=torch.long, device=lidar_bev.device)
            padding = torch.zeros((self.views, max_visible), dtype=torch.bool, device=lidar_bev.device)
            for view in range(self.views):
                index, count = torch.nonzero(visible_mask[sample, view], as_tuple=False).flatten(), int(counts[view].item())
                padding[view, count:] = True
                if count == 0:
                    continue
                indices[view, :count] = index
                grid = reference_points[sample, view, index][None, None] * 2.0 - 1.0
                sampled = [sample_features_by_grid_v2(
                    feature[sample:sample + 1, view:view + 1], grid, None, False, "border"
                )[0, 0] for feature in projected_images]
                query_feature[view, :, :count] = lidar_flat[sample, :, index] + torch.stack(sampled).amax(0)
                query_position[view, :count] = query_positions[sample, view, index]
                refs[view, :count] = reference_points[sample, view, index, None].expand(-1, self.levels, -1)
            query_features.append(query_feature); positions_out.append(query_position)
            refs_out.append(refs); indices_out.append(indices); padding_out.append(padding)
        return {"projected_lidar_bev": projected_lidar, "image_stack": image_stack,
                "query_feature": query_features, "query_position": positions_out,
                "reference_points": refs_out, "bev_indices": indices_out,
                "query_padding_mask": padding_out, "normal_image_positions": normal_image_positions,
                "spatial_shapes": spatial_shapes, "level_start_index": level_start_index,
                "input_padding_mask": input_padding_mask}
