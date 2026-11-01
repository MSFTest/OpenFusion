# -*- coding: utf-8 -*-
"""SparseFusion 最终 sparse LiDAR/image query 融合预测 block。"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.query_transformer_decoder_layer import QueryTransformerDecoderLayer
from common.sparsefusion_layers import ProjectionLayerNorm, SparseFusionConvPredictionHeads, initialize_like_sparsefusion
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class SparseFusionSparseQueryFusionPredictor(nn.Module):
    """拼接 LiDAR 与 image sparse query 后执行全局融合 attention。

    原源码 ``sparsefusion_models.py::FusionTransformer2D_3D_Self``。正式输入
    两路各200个 ``[B,128,P]`` query；两模态先经独立 Linear+LayerNorm，按
    LiDAR-first 拼接为400 query。每层同一个 learned position 实例同时作为
    query/key编码器，Q/K/V均为全部 query，随后预测完整3D框并更新center。
    """

    def __init__(self, hidden_channels=128, num_classes=10, heads=8,
                 ffn_channels=256, dropout=0.1, layers=1):
        super().__init__()
        self.pts_projection, self.img_projection = ProjectionLayerNorm(hidden_channels), ProjectionLayerNorm(hidden_channels)
        self.decoder = nn.ModuleList()
        for _ in range(layers):
            shared = LearnedPositionEmbedding1d(2, hidden_channels, True)
            self.decoder.append(QueryTransformerDecoderLayer(
                hidden_channels, heads, ffn_channels, dropout,
                query_position=shared, key_position=shared, cross_only=True,
            ))
        spec = {"center": (2, 2), "height": (1, 2), "dim": (3, 2),
                "rot": (2, 2), "vel": (2, 2), "heatmap": (num_classes, 2)}
        self.prediction_heads = SparseFusionConvPredictionHeads(hidden_channels, spec)
        initialize_like_sparsefusion(self)

    def forward(self, lidar_feat, lidar_pos, image_feat, image_pos) -> Dict[str, object]:
        all_feat = feature_concatenation([self.pts_projection(lidar_feat), self.img_projection(image_feat)], 2)
        all_pos = feature_concatenation([lidar_pos, image_pos], 1)
        predictions: List[Dict[str, torch.Tensor]] = []
        for decoder in self.decoder:
            all_feat = decoder(all_feat, all_feat, all_pos, all_pos)
            result = self.prediction_heads(all_feat)
            result["center"] = result["center"] + all_pos.transpose(1, 2)
            predictions.append(result)
            all_pos = result["center"].detach().clone().transpose(1, 2)
        return {"query_feat": all_feat, "query_pos": all_pos, "predictions": predictions}
