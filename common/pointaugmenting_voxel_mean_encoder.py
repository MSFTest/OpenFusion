"""体素均值编码。Args: features[Nv,T,69]浮点，num_voxels[Nv]有效点数，同device。Returns: 连续[Nv,69]；坐标不变，无状态参数。
原源码：det3d/models/readers/voxel_encoder.py。参数、buffer及计算顺序按原节点迁移。
"""
from torch import nn
class VoxelFeatureExtractorV3(nn.Module):
    """原voxel_encoder.py::VoxelFeatureExtractorV3；Args为[Nv,T,69]及有效点数，Returns为同device连续[Nv,69]。无参数，无坐标变换。"""
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"
    ):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        assert self.num_input_features == features.shape[-1]

        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()

