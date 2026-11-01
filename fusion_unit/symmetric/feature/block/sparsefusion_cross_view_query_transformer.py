# -*- coding: utf-8 -*-
"""SparseFusion 相机参数调制与跨视图 query transformation block。"""

from __future__ import annotations

import torch
from torch import nn

from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.query_transformer_decoder_layer import QueryTransformerDecoderLayer
from common.sparsefusion_layers import ProjectionLayerNorm, SparseFusionConvPredictionHeads, initialize_like_sparsefusion
from utils.sparsefusion_view_position_descriptor import sparsefusion_view_position_descriptor


class SparseFusionCameraSE(nn.Module):
    """BN(16) 后三层 1×1 Conv 生成 sigmoid camera gate。"""

    def __init__(self, hidden_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(16)
        self.mlp_depth = nn.Sequential(nn.Conv1d(16, hidden_channels, 1), nn.ReLU(inplace=True),
                                       nn.Conv1d(hidden_channels, hidden_channels, 1), nn.ReLU(inplace=True),
                                       nn.Conv1d(hidden_channels, hidden_channels, 1))

    def forward(self, feature, camera_info):
        return feature * self.mlp_depth(self.bn(camera_info)).sigmoid()


class SparseFusionCrossViewQueryTransformer(nn.Module):
    """把相机 query 转换为统一 LiDAR-view query 并跨相机交互。

    原源码 ``sparsefusion_models.py::CameraSE/ViewTransformer``。输入 feature
    ``[B,C,P]``、LiDAR geometry ``[B,P,10]``、camera info ``[B,16,P]``、
    view position ``[B,P,9]``。Q/K/V 均为全部相机 proposals，执行一层全局
    attention；保留 mono旁路字段、view预测和 center residual。
    """

    def __init__(self, hidden_channels=128, num_classes=10, heads=8, ffn_channels=256, dropout=0.1):
        super().__init__()
        self.camera_net = SparseFusionCameraSE(hidden_channels)
        self.view_projection = ProjectionLayerNorm(hidden_channels)
        self.decoder = QueryTransformerDecoderLayer(
            hidden_channels, heads, ffn_channels, dropout,
            query_position=LearnedPositionEmbedding1d(9, hidden_channels, False),
            key_position=LearnedPositionEmbedding1d(9, hidden_channels, False), cross_only=True,
        )
        spec = {"center_view": (2, 2), "height_view": (1, 2), "dim_view": (3, 2),
                "rot_view": (2, 2), "vel_view": (2, 2), "heatmap_view": (num_classes, 2)}
        self.prediction_heads = SparseFusionConvPredictionHeads(hidden_channels, spec)
        initialize_like_sparsefusion(self)

    def forward(self, query_feat, lidar_geometry, camera_info, view_position=None):
        if view_position is None:
            raise ValueError("view_position 必须由 sparsefusion_view_position_descriptor 生成")
        query_feat = self.camera_net(query_feat, camera_info)
        query_feat = self.view_projection(query_feat)
        query_feat = self.decoder(query_feat, query_feat, view_position, view_position)
        result = self.prediction_heads(query_feat)
        for target, start, end in (("center_mono", 0, 2), ("height_mono", 2, 3),
                                   ("rot_mono", 3, 5), ("vel_mono", 5, 7), ("dim_mono", 7, 10)):
            result[target] = lidar_geometry[..., start:end].transpose(1, 2)
        result["center_view"] = result["center_view"] + result["center_mono"]
        return {"query_feat": query_feat, "query_pos": result["center_view"].detach().clone().transpose(1, 2),
                "predictions": [result]}

    @staticmethod
    def build_view_position(lidar_geometry, normal_image_position):
        return sparsefusion_view_position_descriptor(lidar_geometry, normal_image_position)
