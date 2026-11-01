# -*- coding: utf-8 -*-
"""SparseFusion deformable semantic transfer 与多视图聚合 block。"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.sparsefusion_layers import ConvLN, SparseFusionDeformableTransformerDecoderLayer, initialize_like_sparsefusion
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class SparseFusionSemanticDeformableTransfer(nn.Module):
    """将多视角图像语义转移至 BEV 并生成融合 heatmap。

    原源码 ``sparsefusion_head_deform.py:704-760``。同一 BEV 格的多相机结果
    逐元素 maximum，未可见位置回退投影 LiDAR feature；再拼接 0/1 visible
    flag，经 3×3 ConvLN-ReLU 和分类卷积。正式结构 C128/H8/L4/P4/FFN256。
    """

    def __init__(self, hidden_channels=128, num_classes=10, heads=8, levels=4,
                 points=4, ffn_channels=256, dropout=0.1):
        super().__init__()
        self.cross_heatmap_decoder = SparseFusionDeformableTransformerDecoderLayer(
            hidden_channels, heads, levels, ffn_channels, dropout,
            LearnedPositionEmbedding1d(3, hidden_channels, False),
            LearnedPositionEmbedding1d(2, hidden_channels, False), points=points,
        )
        self.reduce_conv = ConvLN(hidden_channels + 1, hidden_channels, 3, padding=1)
        self.cross_heatmap_head = nn.Sequential(nn.Conv2d(hidden_channels, num_classes, 3, padding=1, bias=True))
        initialize_like_sparsefusion(self)

    def forward(self, sampled: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        projected = sampled["projected_lidar_bev"]
        batch, channels, height, width = projected.shape
        output = torch.zeros((batch, channels, height * width), device=projected.device)
        count_map = torch.zeros((batch, 1, height * width), device=projected.device)
        views = sampled["image_stack"].shape[1]
        for sample in range(batch):
            query, position = sampled["query_feature"][sample], sampled["query_position"][sample]
            refs, padding = sampled["reference_points"][sample], sampled["query_padding_mask"][sample]
            input_padding = sampled["input_padding_mask"]
            if batch == 1:
                transformed_all = self.cross_heatmap_decoder(
                    query, sampled["image_stack"][sample], position,
                    sampled["normal_image_positions"].repeat(views, 1, 1), refs,
                    sampled["level_start_index"], sampled["spatial_shapes"], padding,
                    None if input_padding is None else input_padding[sample],
                )
            else:
                transformed_views = []
                for view in range(views):
                    valid = ~padding[view]
                    item = self.cross_heatmap_decoder(
                        query[view, :, valid][None], sampled["image_stack"][sample, view, None],
                        position[view, valid][None], sampled["normal_image_positions"], refs[view, valid][None],
                        sampled["level_start_index"], sampled["spatial_shapes"],
                        input_padding_mask=None if input_padding is None else input_padding[sample, view, None],
                    )
                    padded = item.new_zeros((item.shape[1], padding.shape[1])); padded[:, valid] = item[0]
                    transformed_views.append(padded)
                transformed_all = torch.stack(transformed_views)
            for view in range(views):
                valid, count = ~padding[view], int((~padding[view]).sum().item())
                index = sampled["bev_indices"][sample][view, :count]
                overlap = count_map[sample, 0, index] > 0
                transformed, current = transformed_all[view, :, :count], output[sample, :, index]
                current[:, overlap] = torch.maximum(current[:, overlap], transformed[:, overlap])
                current[:, ~overlap] = transformed[:, ~overlap]
                output[sample, :, index] = current
                count_map[sample, :, index] += 1
        output, count_map = output.reshape(batch, channels, height, width), count_map.reshape(batch, 1, height, width)
        visible = (count_map > 0).to(output.dtype)
        output = output + (1.0 - visible) * projected
        semantic = self.reduce_conv(feature_concatenation([output, visible], 1))
        return {"semantic_bev": semantic, "cross_heatmap_logits": self.cross_heatmap_head(semantic.contiguous()),
                "visible_flag": visible}
