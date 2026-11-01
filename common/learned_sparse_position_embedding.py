# -*- coding: utf-8 -*-
"""UniTR 的稀疏窗口坐标 learned embedding。"""

from torch import nn


class LearnedSparsePositionEmbedding(nn.Module):
    """将窗口坐标按原 UniTR Linear MLP 编码为位置特征。

    原源码：``pcdet/models/model_utils/dsvt_utils.py:10-24``；调用位于
    ``pcdet/models/backbones_3d/dsvt.py:580-615``，抽取版本位于
    ``past/UniTR Module extraction/Stage01_Intra_Modal_Encoding/network/learned_sparse_position_embedding.py``。

    Args:
        input_channel: 坐标维度，常用 2 或 3。
        num_pos_feats: 隐藏层及输出特征维度，默认 128。
    Inputs:
        location: ``[N, input_channel]``，任意 Linear 可读的二维布局；dtype
            与模块参数兼容，device 与模块相同。
    Returns:
        ``[N, num_pos_feats]``，保持输入的 dtype/device。参数及 BatchNorm
        running state 的键与原 ``position_embedding_head`` checkpoint 相同。
    """

    def __init__(self, input_channel, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, location):
        """执行 Linear-BN-ReLU-Linear 坐标编码。

        Args:
            location: ``[N,input_channel]`` 二维张量，支持非连续布局；dtype
                须与模块参数兼容，device 须与模块相同。
        Returns:
            ``[N,num_pos_feats]`` 张量，dtype/device 跟随输入；不改变输入，
            BN running state 按模块 train/eval 状态更新。
        原始实现位置：``dsvt_utils.py:10-24``。
        """
        return self.position_embedding_head(location)
