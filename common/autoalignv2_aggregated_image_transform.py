from torch import nn
from torch.nn import functional as F

from common.linear_projection import linear_projection


class AutoAlignV2AggregatedImageTransform(nn.Module):
    """原聚合图像 Linear/BN 及训练期 dropout，无多源融合。

    来源：multi_voxel_deformfusion.py::__init__ 194–197、forward 268–270。
    Args:
        mid_channels/out_channels: 原正式配置均为 64；dropout_ratio 默认 0。
        forward(image_features): 同参数 dtype/device 浮点 [M,mid_channels]。
    Returns:
        同 dtype/device [M,out_channels]，保持 voxel 原行序，不涉及坐标/box。
        Linear 含 bias；BN1d eps=1e-3、momentum=0.01、affine/running stats 开启。
        原构造初始化、img_transform.0/1 参数及 BN buffers 不变；全 batch 合并后
        调用一次，train/eval 保持 BN/dropout 分支。空输入及单行训练遵循原 BN 行为。
    """

    def __init__(self, mid_channels=64, out_channels=64, dropout_ratio=0):
        super().__init__()
        self.img_transform = nn.Sequential(nn.Linear(mid_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))
        self.dropout_ratio = dropout_ratio

    def forward(self, image_features):
        layer = self.img_transform[0]
        output = self.img_transform[1](linear_projection(image_features, layer.weight, layer.bias))
        if self.training and self.dropout_ratio > 0:
            output = F.dropout(output, self.dropout_ratio)
        return output
