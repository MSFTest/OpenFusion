# -*- coding: utf-8 -*-
"""DeepInteraction 六分支 query 预测头。"""

from __future__ import annotations

from torch import nn

from common.bevdet_lidar_regression_head import LidarOnlyRegressionHead


class PredictionHead(nn.Module):
    """原 SeparateHead 的五个回归分支和一个分类分支。

    Args:
        in_channels: query 输入通道，初始层 128、MMPI 层 256。
        num_classes: nuScenes 类别数，正式配置 10。
    Returns:
        输入 ``[B,C,K]``，返回 center/height/dim/rot/vel/heatmap 字典，
        shape 分别为 ``[B,2/1/3/2/2/num_classes,K]``。

    每个分支严格为 ``Conv1d(C,64,bias=False)-BN-ReLU-Conv1d``。五个回归
    分支复用 ``LidarOnlyRegressionHead`` 的真实子 head，但这里不注入 center
    残差；残差仍由原 decoder/MMPI 调用者执行。原源码：
    ``decoder_utils.py:498-581``。
    """
    def __init__(self, in_channels=128, num_classes=10):
        super().__init__()
        self.regression = LidarOnlyRegressionHead(
            in_channels, num_layers=2, head_channels=64
        )
        self.heatmap = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.Conv1d(64, num_classes, 1),
        )

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射 past 简化键及原 mmcv ``ConvModule`` SeparateHead 键。"""
        prefix = {
            "center": "regression.center_head.layers",
            "height": "regression.height_head.layers",
            "dim": "regression.dim_head.layers",
            "rot": "regression.rot_head.layers",
            "vel": "regression.vel_head.layers",
            "heatmap": "heatmap",
        }
        mmcv_style = any(
            ".0.conv." in key or ".0.bn." in key for key in state_dict
        )
        mapped = {}
        for key, value in state_dict.items():
            field, dot, rest = key.partition(".")
            if rest.startswith("0.conv."):
                rest = "0." + rest[len("0.conv."):]
            elif rest.startswith("0.bn."):
                rest = "1." + rest[len("0.bn."):]
            elif mmcv_style and rest.startswith("1."):
                rest = "3." + rest[len("1."):]
            mapped[f"{prefix[field]}.{rest}" if dot else prefix[field]] = value
        return mapped

    def forward(self, feature):
        regression = self.regression
        return {
            "center": regression.center_head(feature),
            "height": regression.height_head(feature),
            "dim": regression.dim_head(feature),
            "rot": regression.rot_head(feature),
            "vel": regression.vel_head(feature),
            "heatmap": self.heatmap(feature),
        }
