"""GMA 门控。原源码：mmdet3d/models/middle_encoders/
sparse_multimodal_encoder_painting.py，SparseMultiModalEncoderPaint.
make_grouped_sparse_conv_blocks 126–176、grouped_sparse_conv 371–401 行。
"""

from __future__ import annotations

import torch
from torch import nn
from fusion_unit.symmetric.feature.ops.feature_gate_multiply import feature_gate_multiply


class GatedModalitySelection(nn.Module):
    """四尺度共享一份定义，各尺度各自有两个 Linear+ReLU 参数。

    Args: LiDAR 特征 [M,C3]；混合行对应 3D/2D 特征 [K,C3]/[K,64]；
    相机独有特征 [Q,64] 与参考行号 [Q]。Returns: 门控后的相机独有
    与混合特征，均 [*,64]。原源码对未匹配的 -1 索引以每次 forward
    新生成的 rand(1,C3) 作为 dummy，不设 buffer、不可学习；保留这一
    随机行为。原模块 state_dict 前缀 gate_control/cross_gate_control
    在本模块中不变，装载时映射至原 multimodal_middle_encoder 同名键。
    """

    def __init__(self, in_channels_3d=(16, 32, 64, 128), in_channels_2d=(64, 64, 64, 64)):
        super().__init__()
        self.gate_control = nn.ModuleList([
            nn.Sequential(nn.Linear(c3, c2), nn.ReLU())
            for c3, c2 in zip(in_channels_3d, in_channels_2d)
        ])
        self.cross_gate_control = nn.ModuleList([
            nn.Sequential(nn.Linear(c3, c2), nn.ReLU())
            for c3, c2 in zip(in_channels_3d, in_channels_2d)
        ])

    def forward(self, lidar_features: torch.Tensor, mixed_lidar_features: torch.Tensor,
                mixed_camera_features: torch.Tensor, only_camera_features: torch.Tensor,
                reference_indices: torch.Tensor, stage_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """原源码 371–401 行；mixed 来源的同步索引由 F3 输出。"""
        dummy_embedding = torch.rand(1, lidar_features.shape[1]).to(lidar_features.device)
        cross_input = torch.cat([lidar_features, dummy_embedding], dim=0)
        cross_gating = self.cross_gate_control[stage_id](cross_input)
        gated_only_camera = feature_gate_multiply(only_camera_features, cross_gating[reference_indices], False)
        gated_mixed_camera = feature_gate_multiply(mixed_camera_features, self.gate_control[stage_id](mixed_lidar_features), False)
        return gated_only_camera, gated_mixed_camera
