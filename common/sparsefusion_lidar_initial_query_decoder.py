# -*- coding: utf-8 -*-
"""SparseFusion 初始 LiDAR query decoder。"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.query_transformer_decoder_layer import QueryTransformerDecoderLayer
from common.sparsefusion_layers import SparseFusionConvPredictionHeads, initialize_like_sparsefusion


class SparseFusionLiDARInitialQueryDecoder(nn.Module):
    """以原始 LiDAR BEV memory 解码初始 query。

    原源码 ``sparsefusion_models.py::PointTransformer2D_3D``。正式输入 query
    ``[B,128,200]``、position ``[B,200,2]``、BEV ``[B,128,H,W]``；输出
    更新 query/position 与逐层3D框预测。decoder 前的初始预测必须执行，以
    保留训练态共享预测头 BN running statistics 副作用。
    """

    def __init__(self, hidden_channels=128, num_classes=10, heads=8,
                 ffn_channels=256, dropout=0.1, layers=1):
        super().__init__()
        spec = {"center": (2, 2), "height": (1, 2), "dim": (3, 2),
                "rot": (2, 2), "vel": (2, 2), "heatmap": (num_classes, 2)}
        self.prediction_heads = SparseFusionConvPredictionHeads(hidden_channels, spec)
        self.decoder = nn.ModuleList([
            QueryTransformerDecoderLayer(
                hidden_channels, heads, ffn_channels, dropout,
                query_position=LearnedPositionEmbedding1d(2, hidden_channels, True),
                key_position=LearnedPositionEmbedding1d(2, hidden_channels, True),
            ) for _ in range(layers)
        ])
        initialize_like_sparsefusion(self)

    def forward(self, query_feat, query_pos, lidar_bev, bev_pos) -> Dict[str, object]:
        key, predictions = lidar_bev.flatten(2), []
        initial_result = self.prediction_heads(query_feat)
        initial_result["center"] = query_pos.transpose(1, 2)
        for decoder in self.decoder:
            query_feat = decoder(query_feat, key, query_pos, bev_pos)
            result = self.prediction_heads(query_feat)
            result["center"] = result["center"] + query_pos.transpose(1, 2)
            predictions.append(result)
            query_pos = result["center"].detach().clone().transpose(1, 2)
        return {"query_feat": query_feat, "query_pos": query_pos, "predictions": predictions}
