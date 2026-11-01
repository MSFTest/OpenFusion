# -*- coding: utf-8 -*-
"""UniTR 最近投影点偏移的位置 embedding。"""

from torch import nn


class NearestProjectionDistanceEmbedding(nn.Module):
    """把最近投影点的二维偏移编码后原位加到目标位置上。

    原源码：``pcdet/models/mm_backbone/unitr.py:123-124, 296, 300-303``；
    基础 embedding 定义见 ``pcdet/models/model_utils/dsvt_utils.py:10-24``，
    抽取版本位于
    ``past/UniTR Module extraction/Stage03_3D_Cross_Modal_Fusion/network/nearest_projection_distance_embedding.py``。

    Args:
        num_pos_feats: 位置特征维度，默认 128。
    Inputs:
        nearest_offset: ``[N,2]`` 最近投影点像素偏移。
        target_pos: ``[N,num_pos_feats]`` 目标位置特征；执行原位 ``+=``，
            因此返回张量与该参数是同一个对象、共享 storage。
    Returns:
        ``target_pos`` 本身，shape/dtype/device/layout 均由输入保持；模块
        参数及 BatchNorm 状态键兼容原 ``position_embedding_head`` checkpoint。
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(2, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, nearest_offset, target_pos):
        """编码偏移并原位累加到目标位置。

        Args:
            nearest_offset: ``[N,2]`` 二维偏移，支持非连续布局；dtype/device
                须与模块参数兼容。
            target_pos: ``[N,num_pos_feats]`` 目标位置，dtype/device/layout
                须可原位接收编码结果。
        Returns:
            返回同一 ``target_pos`` 对象和 storage；shape/dtype/device/layout
            保持不变，版本计数增加一次。BN running state 按 train/eval 更新。
        原始调用位置：``unitr.py:296, 300-303``。
        """
        target_pos += self.position_embedding_head(nearest_offset)
        return target_pos
