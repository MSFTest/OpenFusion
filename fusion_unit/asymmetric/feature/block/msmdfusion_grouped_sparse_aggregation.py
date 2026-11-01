"""GMA 分组与统一稀疏卷积。原源码：mmdet3d/models/middle_encoders/
sparse_multimodal_encoder_painting.py，SparseMultiModalEncoderPaint.
make_grouped_sparse_conv_blocks 126–176、make_aggregation_block 178–189、
grouped_sparse_conv 325–430 行；mmdet3d/ops/sparse_block.py，
SparseBasicBlock 68–126、make_sparse_convmodule 129–191 行。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from common.msmdfusion_sparse_layers import SparseTensor, SparseConvNormAct, SparseBasicBlockReference
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class GroupedSparseAggregation(nn.Module):
    """四尺度同构节点，各尺度独立参数；索引为 (batch,mix,z,y,x)。

    原构造中 3D、2D、mixed 三组子网全部建立，但源码 forward 仅调用
    3D 子网与 aggregation_blocks；2D/mixed 权重仍保留以匹配原
    state_dict。稀疏卷积采用私有数学等价 PyTorch 参考，权重形状
    [out,kz,ky,kx,in]、BN1d 统计与原 spconv2 一致。

    Args: in_channels_3d 为四尺度 LiDAR 通道；smoke_reference 只允许
        显式对照路径。forward 输入 LiDAR/相机 SparseTensor、同步行号、
        相机门控特征及 stage_id。
    Returns: SparseTensor，特征 [M,C3+64]，坐标为 int32 [M,4]
        (batch,z,y,x)，空间尺寸及 batch_size 沿用相机稀疏张量。
    """

    def __init__(self, in_channels_3d=(16, 32, 64, 128), *, smoke_reference=False):
        super().__init__()
        self.backend = "reference" if smoke_reference else "mmcv"
        self.in_channels_3D = in_channels_3d
        self.grouped_sp_conv_blocks_3D = nn.ModuleDict()
        self.grouped_sp_conv_blocks_2D = nn.ModuleDict()
        self.grouped_sp_conv_blocks_mix = nn.ModuleDict()
        self.aggregation_blocks = nn.ModuleDict()
        for i, c in enumerate(in_channels_3d):
            stage = f'stage_{i + 1}'
            self.grouped_sp_conv_blocks_3D[stage] = SparseConvNormAct(c, c, smoke_reference=smoke_reference)
            self.grouped_sp_conv_blocks_2D[stage] = SparseConvNormAct(64, 64, smoke_reference=smoke_reference)
            self.grouped_sp_conv_blocks_mix[stage] = SparseBasicBlockReference(c + 64, smoke_reference=smoke_reference)
            self.aggregation_blocks[stage] = SparseBasicBlockReference(c + 64, smoke_reference=smoke_reference)
        # 原 grouped_sparse_conv 仅调用 3D 与 aggregation 子网；2D/mix
        # 参数保留以兼容原 state_dict，但其原训练轨迹始终无梯度。
        self.grouped_sp_conv_blocks_2D.requires_grad_(False)
        self.grouped_sp_conv_blocks_mix.requires_grad_(False)

    def forward(self, lidar: SparseTensor, camera: SparseTensor,
                syn_mix_3d: torch.Tensor, syn_mix_2d: torch.Tensor,
                only_camera_indices: torch.Tensor, gated_only_camera: torch.Tensor,
                gated_mixed_camera: torch.Tensor, stage_id: int) -> SparseTensor:
        """原 grouped_sparse_conv 379–430 行，不执行未调用的 2D/mix 卷积。"""
        only_3d = lidar.indices[:, 1] == 0
        lidar_only = SparseTensor(lidar.features[only_3d],
                                  lidar.indices[only_3d][:, [0, 2, 3, 4]],
                                  lidar.spatial_shape, lidar.batch_size)
        camera_only = SparseTensor(gated_only_camera, only_camera_indices[:, [0, 2, 3, 4]],
                                   camera.spatial_shape, camera.batch_size)
        mixed_lidar = lidar.features[syn_mix_3d]
        assert mixed_lidar.shape[0] == gated_mixed_camera.shape[0]
        mixed_features = feature_concatenation([mixed_lidar, gated_mixed_camera], -1)
        mixed_indices = camera.indices[syn_mix_2d]
        present = mixed_indices[:, 0].unique().cpu()
        for batch_id in range(lidar.batch_size):
            if batch_id not in present:
                pad_indices = torch.zeros_like(lidar.indices[0]).unsqueeze(0)
                pad_indices[0, 0] = batch_id
                pad_features = torch.zeros_like(torch.cat([lidar.features[0], camera.features[0]])).unsqueeze(0)
                mixed_indices = torch.cat([mixed_indices, pad_indices], dim=0)
                mixed_features = torch.cat([mixed_features, pad_features], dim=0)
        mixed = SparseTensor(mixed_features, mixed_indices[:, [0, 2, 3, 4]],
                             camera.spatial_shape, camera.batch_size)

        stage = f'stage_{stage_id + 1}'
        lidar_only = self.grouped_sp_conv_blocks_3D[stage](lidar_only)
        camera_only = camera_only.replace_feature(F.pad(camera_only.features, (self.in_channels_3D[stage_id], 0)))
        lidar_only = lidar_only.replace_feature(F.pad(lidar_only.features, (0, 64)))
        assert lidar_only.features.shape[1] == camera_only.features.shape[1] == mixed.features.shape[1]
        unified = SparseTensor(
            feature_concatenation([lidar_only.features, camera_only.features, mixed.features], 0),
            torch.cat([lidar_only.indices, camera_only.indices, mixed.indices], dim=0),
            mixed.spatial_shape, camera.batch_size)
        return self.aggregation_blocks[stage](unified)
