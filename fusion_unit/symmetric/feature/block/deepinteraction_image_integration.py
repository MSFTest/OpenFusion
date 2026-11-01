# -*- coding: utf-8 -*-
"""DeepInteraction 图像分支跨模态特征整合块。"""

from torch import nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class ImageIntegration(nn.Module):
    """两级 1x1 Conv-BN 整合 P2I、I2I 与旧图像特征。

    输入/输出为 ``[B,V,C,H,W]``，concat 顺序严格为
    ``[lidar_to_image,image_to_image]``、``[augmented,old_image]``，两层无
    activation，正式 C=128。原源码：``deepinteraction_encoder.py:17-18,32-33``。
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
            key = key.replace("I_out_proj.", "out_projection.", 1)
            key = key.replace("I_integration.", "integration.", 1)
            key = (key.replace(".conv.", ".block.0.")
                   .replace(".norm.", ".block.1.")
                   .replace(".bn.", ".block.1."))
            mapped[key] = value
        return mapped

    def forward(self, lidar_to_image, image_to_image, old_image):
        batch, views, channels, height, width = old_image.shape
        augmented = self.out_projection(feature_concatenation(
            [lidar_to_image, image_to_image], 2
        ).reshape(batch * views, 2 * channels, height, width))
        return self.integration(feature_concatenation([
            augmented, old_image.reshape(batch * views, channels, height, width)
        ], 1)).reshape(batch, views, channels, height, width)
