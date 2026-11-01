# -*- coding: utf-8 -*-
"""DeepInteraction 初始 LiDAR query 解码层。"""

from torch import nn

from common.deepinteraction_prediction_head import PredictionHead
from common.deepinteraction_transformer_decoder_layer import TransformerDecoderLayer


class InitialLiDARQueryDecoder(nn.Module):
    """以融合前 LiDAR memory 解码初始 query 并生成第一层预测。

    Args:
        channels/heads/feedforward/dropout/num_classes: 与原正式配置一致，
        默认 ``128/8/256/.1/10``。
    Returns:
        接收初始化 state；返回补入 query、detached query_position、
        initial_prediction 和 prediction 的 state。输入输出 token 均为
        ``[B,C,K]``，位置为 ``[B,K,2]``。

    注意原可执行源码使用 representational interaction 前的
    ``pre_fusion_flat`` 作为首层 Transformer memory；post-fusion 特征只用于
    第二 heatmap 和后续 LiDAR MMPI。原源码：
    ``deepinteraction_decoder.py:255-270``。
    """
    def __init__(self, channels=128, heads=8, feedforward=256,
                 dropout=.1, num_classes=10):
        super().__init__()
        self.decoder = TransformerDecoderLayer(channels, heads, feedforward, dropout)
        self.prediction = PredictionHead(channels, num_classes)
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def remap_original_component_state_dicts(decoder_state_dict,
                                             prediction_state_dict):
        """合并映射原 ``decoder[0]`` 与 ``prediction_heads[0]`` 权重。"""
        decoder = TransformerDecoderLayer.remap_original_state_dict(
            decoder_state_dict
        )
        prediction = PredictionHead.remap_original_state_dict(
            prediction_state_dict
        )
        return {
            **{"decoder." + key: value for key, value in decoder.items()},
            **{"prediction." + key: value for key, value in prediction.items()},
        }

    def forward(self, state):
        query = self.decoder(
            state["query"], state["pre_fusion_flat"],
            state["query_position"], state["bev_position"],
        )
        prediction = self.prediction(query)
        prediction["center"] = (
            prediction["center"] + state["query_position"].permute(0, 2, 1)
        )
        result = dict(state)
        result.update(
            query=query,
            query_position=prediction["center"].detach().clone().permute(0, 2, 1),
            initial_prediction=prediction,
            prediction=prediction,
        )
        return result
