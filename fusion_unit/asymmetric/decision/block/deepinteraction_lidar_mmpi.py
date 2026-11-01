# -*- coding: utf-8 -*-
"""DeepInteraction LiDAR 侧多模态预测交互块。"""

import torch
from torch import nn
import torch.nn.functional as F

from common.deepinteraction_prediction_head import PredictionHead
from fusion_unit.asymmetric.decision.block.deepinteraction_dynamic_conv import DynamicConv
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class LiDARMMPI(nn.Module):
    """以 LiDAR ROI 更新全部 query 并预测残差。

    Args:
        channels/heads/dropout/roi_size/num_classes: 正式值为
        ``128/8/.1/7/10``。
    Returns:
        query ``[B,C,K]``、上一层预测和 ROI ``[B,K,C,S,S]``，返回更新 query
        和六分支预测；center 对上一层 detached center 做残差，无可见性回退。
    原源码：``decoder_utils.py:765-841`` 和
    ``deepinteraction_decoder.py:281-296``。
    """
    def __init__(self, channels=128, heads=8, dropout=.1,
                 roi_size=7, num_classes=10):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(channels, heads, dropout=dropout)
        self.dynamic_conv = DynamicConv(channels, roi_size)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, channels * 4)
        self.linear2 = nn.Linear(channels * 4, channels)
        self.dropout = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(channels)
        self.prediction = PredictionHead(2 * channels, num_classes)

    @staticmethod
    def remap_original_block_state_dict(state_dict):
        """映射原 ``PointRCNNBlock`` 的可训练层（预测头在原源码中独立）。"""
        replacements = {
            "dyconv_pre_self_attn_pts.": "self_attention.",
            "dyconv_pts.": "dynamic_conv.",
            "norm1_pts.": "norm1.", "norm2_pts.": "norm2.",
            "norm3_pts.": "norm3.", "linear1_pts.": "linear1.",
            "linear2_pts.": "linear2.",
        }
        mapped = {}
        for key, value in state_dict.items():
            for old, new in replacements.items():
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            mapped[key] = value
        return mapped

    def forward(self, previous_query, previous_prediction, roi_feature):
        batch = roi_feature.shape[0]
        query = torch.zeros_like(previous_query)
        for sample in range(batch):
            current = previous_query[sample].transpose(0, 1).unsqueeze(1)
            update = self.self_attention(current, current, current)[0]
            current = self.norm1(current + self.dropout1(update)).transpose(0, 1)
            roi = roi_feature[sample].flatten(2).permute(2, 0, 1)
            update = self.dynamic_conv(current, roi)
            current = self.norm2(current + self.dropout2(update))
            update = self.linear2(self.dropout(F.gelu(self.linear1(current))))
            current = self.norm3(current + self.dropout3(update))
            query[sample] = current[0].transpose(0, 1)
        prediction = self.prediction(feature_concatenation(
            [query, previous_query], 1
        ))
        prediction["center"] = (
            prediction["center"] + previous_prediction["center"].detach().clone()
        )
        return query, prediction
