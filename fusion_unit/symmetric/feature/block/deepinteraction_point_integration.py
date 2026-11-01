# -*- coding: utf-8 -*-
"""DeepInteraction 点分支跨模态特征整合块。"""

from torch import nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class PointIntegration(nn.Module):
    """两级 1x1 Conv-BN 整合 I2P、P2P 与旧 LiDAR 特征。

    输入均为 ``[B,C,H,W]``，输出同布局；concat 顺序严格为
    ``[image_to_point,point_to_point]``、``[augmented,old_lidar]``。两层均无
    activation，正式 C=128。原源码：``deepinteraction_encoder.py:12-13,27-28``。
    """
    def __init__(self, channels=128):
        super().__init__()
        self.out_projection = Conv2dNormActivationBlock(
            2 * channels, channels, 1, activation=False
        )
        self.integration = Conv2dNormActivationBlock(
            2 * channels, channels, 1, activation=False
        )

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 ``ConvBN.conv/norm`` checkpoint 键到复用 block。"""
        mapped = {}
        for key, value in state_dict.items():
            key = key.replace("P_out_proj.", "out_projection.", 1)
            key = key.replace("P_integration.", "integration.", 1)
            key = (key.replace(".conv.", ".block.0.")
                   .replace(".norm.", ".block.1.")
                   .replace(".bn.", ".block.1."))
            mapped[key] = value
        return mapped

    def forward(self, image_to_point, point_to_point, old_lidar):
        augmented = self.out_projection(feature_concatenation(
            [image_to_point, point_to_point], 1
        ))
        return self.integration(feature_concatenation([augmented, old_lidar], 1))
