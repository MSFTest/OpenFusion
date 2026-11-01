"""多尺度逐级交互。原源码：mmdet3d/models/middle_encoders/
sparse_multimodal_encoder_painting.py，SparseMultiModalEncoderPaint.__init__
101–124、make_downscale_block 191–206、forward 433–459 行；配置
configs/MSMDFusion_nusc_voxel_LC.py 182–190 行。
"""

from __future__ import annotations

from torch import nn

from common.msmdfusion_sparse_layers import (
    SparseTensor, SparseConvNormAct, sparse_add)


class ProgressiveSparseInteraction(nn.Module):
    """四级 sparse_add 后下采样；输入/输出均为 LiDAR (b,z,y,x) 稀疏坐标。

    Stage 1 无前级张量，Stage 2–4 先按 spconv.functional.sparse_add
    的 COO coalesce 语义逐坐标求和，再执行 SparseConv3d/BN1d/ReLU。
    实际参数通道为 80→96、96→128、128→192、192→192；核分别
    3/3/3/(3,1,1)，stride 2/2/2/(2,1,1)，padding 1/1/
    (0,1,1)/0。state_dict 保留原 downscale_blocks.stage_i 前缀。

    Args: 通道、padding、kernel、stride 与原配置一致；smoke_reference
        只允许显式对照路径。forward 接收当前/前级 SparseTensor 和 stage_id。
    Returns: 下采样 SparseTensor，特征 [M,out_channels[stage]+64]，
        int32 坐标布局仍为 (batch,z,y,x)。
    """

    def __init__(self, in_channels_3d=(16, 32, 64, 128), out_channels=(32, 64, 128, 128),
                 padding=(1, 1, (0, 1, 1), 0),
                 down_kernel_size=(3, 3, 3, (3, 1, 1)),
                 down_stride=(2, 2, 2, (2, 1, 1)), *, smoke_reference=False):
        super().__init__()
        self.backend = "reference" if smoke_reference else "mmcv"
        self.downscale_blocks = nn.ModuleDict()
        for i in range(4):
            self.downscale_blocks[f'stage_{i + 1}'] = SparseConvNormAct(
                in_channels_3d[i] + 64, out_channels[i] + 64,
                down_kernel_size[i], down_stride[i], padding[i], subm=False, smoke_reference=smoke_reference)

    def forward(self, current: SparseTensor, previous: SparseTensor | None,
                stage_id: int) -> SparseTensor:
        """原 forward 449–457 行；同一 network 实例调用四次，参数不共享。"""
        if stage_id:
            current = sparse_add(current, previous)
        return self.downscale_blocks[f'stage_{stage_id + 1}'](current)
