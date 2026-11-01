from torch import nn

from common.linear_projection import linear_projection


class AutoAlignV2VoxelFeatureTransform(nn.Module):
    """原 voxel 主分支 Linear/BN，保持梯度及独立参数。

    来源：multi_voxel_deformfusion.py::__init__ 203–206、forward 271。
    Args:
        pts_channels/out_channels: 正式均 64；forward(voxel_features): 与参数同
            dtype/device 的浮点 [M,pts_channels]，不要求连续，不涉及坐标变换。
    Returns:
        同 dtype/device [M,out_channels]，保留原行序及输入梯度，不 detach。
        Linear bias=True；BN1d eps=1e-3、momentum=0.01，保留 affine、running
        buffers、默认构造初始化和 train/eval 行为；无 ReLU/残差/dropout。
        pts_transform.0/1 键与原权重一致。空/单行输入沿用原 Linear/BN 行为。
    """

    def __init__(self, pts_channels=64, out_channels=64):
        super().__init__()
        self.pts_transform = nn.Sequential(nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))

    def forward(self, voxel_features):
        layer = self.pts_transform[0]
        return self.pts_transform[1](linear_projection(voxel_features, layer.weight, layer.bias))
