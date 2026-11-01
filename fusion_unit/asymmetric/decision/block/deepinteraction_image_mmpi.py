# -*- coding: utf-8 -*-
"""DeepInteraction 图像侧多模态预测交互块。"""

import torch
from torch import nn
import torch.nn.functional as F

from common.deepinteraction_prediction_head import PredictionHead
from fusion_unit.asymmetric.decision.block.deepinteraction_dynamic_conv import DynamicConv
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class ImageMMPI(nn.Module):
    """用可见图像 ROI 更新 query，并对不可见 query 回退首层预测。

    Args:
        channels/heads/dropout/roi_size/num_classes: 正式值为
        ``128/8/.1/7/10``。
    Returns:
        输入 query ``[B,C,K]``、三组预测字典、ROI
        ``[B,V,K,C,S,S]``、可见性 ``[B,V,K]``；返回更新 query、预测和
        ``seen[B,K]``。每个相机可见数必须大于 1 才处理；后相机覆盖前相机；
        未见 query 的所有字段回退到第一层 ``initial_prediction``。
    原源码：``decoder_utils.py:632-761`` 和
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
        """映射原 ``ImageRCNNBlock`` 的可训练层（预测头在原源码中独立）。"""
        replacements = {
            "dyconv_pre_self_attn.": "self_attention.",
            "dyconv.": "dynamic_conv.",
        }
        mapped = {}
        for key, value in state_dict.items():
            for old, new in replacements.items():
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            mapped[key] = value
        return mapped

    def forward(self, previous_query, previous_prediction, initial_prediction,
                roi_feature, visible):
        batch, views = roi_feature.shape[:2]
        query = torch.zeros_like(previous_query)
        seen = visible.any(1)
        for sample in range(batch):
            for view in range(views):
                chosen = visible[sample, view]
                if chosen.sum() <= 1:
                    continue
                current = previous_query[sample, :, chosen].transpose(0, 1).unsqueeze(1)
                update = self.self_attention(current, current, current)[0]
                current = self.norm1(current + self.dropout1(update)).transpose(0, 1)
                roi = roi_feature[sample, view, chosen].flatten(2).permute(2, 0, 1)
                update = self.dynamic_conv(current, roi)
                current = self.norm2(current + self.dropout2(update))
                update = self.linear2(self.dropout(F.gelu(self.linear1(current))))
                current = self.norm3(current + self.dropout3(update))
                query[sample, :, chosen] = current[0].transpose(0, 1).clone()
        prediction = self.prediction(feature_concatenation(
            [query, previous_query], 1
        ))
        position = previous_prediction["center"].detach().clone()
        prediction["center"] = prediction["center"] + position
        for field, value in prediction.items():
            mask = ~seen[:, None].expand(-1, value.shape[1], -1)
            value[mask] = initial_prediction[field][mask]
        return query, prediction, seen
