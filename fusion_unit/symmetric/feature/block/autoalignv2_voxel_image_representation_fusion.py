from torch import nn
from torch.nn import functional as F

from common.linear_projection import linear_projection
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class AutoAlignV2VoxelImageRepresentationFusion(nn.Module):
    """原双路表示融合：concat→ReLU→Linear→BN→ReLU。

    来源：multi_voxel_deformfusion.py::__init__ 208–214、forward 274–278；
    对应正式 activate_out=fuse_out=True 分支，不是检测 head。
    Args:
        out_channels: 原默认 64；forward(image_features, voxel_features): 两个
            同参数 dtype/device 浮点 [M,out_channels]，严格按 image/voxel 顺序拼接。
    Returns:
        同 dtype/device [M,out_channels]，不重排体素行，不处理坐标/box。
        Linear(2C,C,bias=True)，BN1d eps=1e-3、momentum=0.01；两个 ReLU 均非 inplace。
        两路及参数均可微；保留 fuse_conv.0/1/2、BN affine/running buffers、默认构造
        初始化及 train/eval 行为，无额外残差。空/单行输入遵循原 BN 边界。
    """

    def __init__(self, out_channels=64):
        super().__init__()
        self.fuse_conv = nn.Sequential(nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU(inplace=False))

    def forward(self, image_features, voxel_features):
        fused = F.relu(feature_concatenation([image_features, voxel_features], -1))
        layer = self.fuse_conv[0]
        return self.fuse_conv[2](self.fuse_conv[1](linear_projection(fused, layer.weight, layer.bias)))
