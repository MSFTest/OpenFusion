# -*- coding: utf-8 -*-
"""SparseFusion 初始多视角图像 query decoder。"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn

from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.sparsefusion_layers import (
    SparseFusionDeformableTransformerDecoderLayer,
    SparseFusionLinearPredictionHeads,
    initialize_like_sparsefusion,
)


class SparseFusionImageInitialQueryDecoder(nn.Module):
    """按 query 来源相机分组读取四尺度图像 memory。

    原源码 ``sparsefusion_models.py::ImageTransformer_Cam_3D_MS``。正式为
    C128、六视图、四尺度、8 heads、每头每层4点、一层 decoder。来源相机
    query 数 ``<=1`` 时严格跳过且更新特征保持零；有效相机在每个样本内一次
    批量执行，以保留训练态 dropout 批次语义。
    """

    def __init__(self, hidden_channels=128, num_classes=10, views=6, levels=4,
                 heads=8, ffn_channels=256, dropout=0.1, points=4, layers=1):
        super().__init__()
        self.views, self.levels = views, levels
        spec = {"center_2d": (2, 3, False), "depth_2d": (1, 3, False),
                "cls": (num_classes, 2), "dim_2d": (3, 3, False),
                "rot_2d": (2, 3, False), "vel_2d": (2, 3, False)}
        self.prediction_heads = SparseFusionLinearPredictionHeads(hidden_channels, spec)
        self.decoder = nn.ModuleList([
            SparseFusionDeformableTransformerDecoderLayer(
                hidden_channels, heads, levels, ffn_channels, dropout,
                LearnedPositionEmbedding1d(2, hidden_channels, False),
                LearnedPositionEmbedding1d(2, hidden_channels, False), points=points,
            ) for _ in range(layers)
        ])
        initialize_like_sparsefusion(self)

    def forward(self, query_feat, normal_query_pos, query_view,
                image_features: Sequence[torch.Tensor], normal_image_positions,
                spatial_shapes, level_start_index, input_padding_mask) -> Dict[str, object]:
        batch, channels, proposals = query_feat.shape
        image_stack = torch.cat([feature.reshape(batch, self.views, channels, -1)
                                 for feature in image_features], -1)
        predictions: List[Dict[str, torch.Tensor]] = []
        for decoder in self.decoder:
            previous, query_feat = query_feat.clone(), torch.zeros_like(query_feat)
            for sample in range(batch):
                counts = torch.bincount(query_view[sample], minlength=self.views)
                view_mask, max_len = counts > 1, int(counts.max().item())
                sample_query = torch.zeros((self.views, channels, max_len), device=query_feat.device)
                sample_pos = torch.zeros((self.views, max_len, 2), device=query_feat.device)
                sample_ref = torch.zeros((self.views, max_len, self.levels, 2), device=query_feat.device)
                sample_padding = torch.zeros((self.views, max_len), dtype=torch.bool, device=query_feat.device)
                for view in range(self.views):
                    selected = query_view[sample] == view
                    count = int(counts[view].item())
                    if count <= 1:
                        continue
                    sample_query[view, :, :count] = previous[sample, :, selected]
                    sample_pos[view, :count] = normal_query_pos[sample, selected]
                    sample_ref[view, :count] = normal_query_pos[sample, selected, None].sigmoid().expand(-1, self.levels, -1)
                    sample_padding[view, count:] = True
                sample_query[view_mask] = decoder(
                    sample_query[view_mask], image_stack[sample, view_mask], sample_pos[view_mask],
                    normal_image_positions.repeat(int(view_mask.sum().item()), 1, 1), sample_ref[view_mask],
                    level_start_index, spatial_shapes, sample_padding[view_mask], input_padding_mask[sample, view_mask],
                )
                for view in range(self.views):
                    selected, count = query_view[sample] == view, int(counts[view].item())
                    if count > 1:
                        query_feat[sample, :, selected] = sample_query[view, :, :count]
            result = self.prediction_heads(query_feat)
            raw_center = result["center_2d"] + normal_query_pos.transpose(1, 2)
            normal_query_pos = raw_center.detach().clone().transpose(1, 2)
            result["center_2d"] = raw_center.sigmoid()
            predictions.append(result)
        return {"query_feat": query_feat, "normal_query_pos": normal_query_pos,
                "query_view": query_view.clone(), "predictions": predictions}
