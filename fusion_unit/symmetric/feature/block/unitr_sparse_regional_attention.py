"""UniTR 稀疏区域注意力 block。

来源：UniTR ``pcdet/models/mm_backbone/unitr.py`` 的 ``SetAttention``、
``UniTR_EncoderLayer``、``UniTRBlock`` 及 backbone block 调用处。保留原两
shift encoder、两套独立参数、LiDAR/image 分支的 FFN 与归一化、最外层
residual/norm、train-only activation checkpoint 和原参数名。

Args:
    d_model: 特征通道数。
    nhead: 每个区域的注意力头数。
    dim_feedforward: FFN 隐藏通道数。
    dropout: 多头注意力 dropout。
    activation: ``gelu`` 或 ``relu``。
    split_ffn: 是否分别为 LiDAR 与 image 使用 FFN/norm。
    split_residual: 是否分别归一化 block 最外层的 LiDAR/image residual。

输入/shape：``src`` 与每个 ``pos_embed_list[shift]`` 为 ``(N, d_model)``；
``set_voxel_inds_list[shift][block_id % 2]`` 与 masks 同为 ``(set_num, set_size)``；
索引为 ``src`` 的有效行号，padding 项须由 mask 标识且仍使用有效索引。
每个 set 至少有一个未 mask token；全 mask set 保留原 MultiheadAttention 行为，
可能产生 NaN。``voxel_num`` 是 ``src`` 中 LiDAR 行数，后续行为 image 行。
两 shift 各执行一次。

Returns:
    ``(N, d_model)`` 输出。Q/K 为 gathered feature+position，V 为 gathered
    feature；注意力输出按原 unique/reverse-scatter 规则回写，再执行原残差与
    modality-specific norm/FFN。

State:
    ``state_dict`` 名称与 past 抽取模块及 ``UniTRBlock.encoder_list`` 一致；
    block 尾部的两组 residual norm 对应 UniTR backbone 中位于 block 外的 norm。
    ``unitr_map.yaml`` 配置 ``split_ffn=True``、``split_residual=True``；原始实现
    也支持两项为 False，当前 block 保留并测试这些分支。
    ``load_original_state_dict`` 可按 block_id 加载裸 backbone 或以
    ``mm_backbone.`` 开头的 detector checkpoint 参数。
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _SetAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8, dim_feedforward=256, dropout=0.0,
                 mlp_dropout=0.0, activation="gelu", split_ffn=True):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.split_ffn = split_ffn
        if split_ffn:
            self.lidar_linear1 = nn.Linear(d_model, dim_feedforward)
            self.lidar_dropout = nn.Dropout(mlp_dropout)
            self.lidar_linear2 = nn.Linear(dim_feedforward, d_model)
            self.lidar_norm1 = nn.LayerNorm(d_model)
            self.lidar_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.d_model = d_model

    def forward(self, src, pos, key_padding_mask, voxel_inds, voxel_num):
        set_features = src[voxel_inds]
        set_pos = pos[voxel_inds]
        query = set_features + set_pos
        key = set_features + set_pos
        value = set_features
        if key_padding_mask is not None:
            src2 = self.self_attn(query, key, value, key_padding_mask=key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]

        flat_inds = voxel_inds.reshape(-1)
        unique_inds, inverse = torch.unique(flat_inds, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique_inds.size(0)).scatter_(0, inverse, perm)
        src2 = src2.reshape(-1, self.d_model)[perm]

        if self.split_ffn:
            src = src + self.dropout1(src2)
            lidar_norm = self.lidar_norm1(src[:voxel_num])
            image_norm = self.norm1(src[voxel_num:])
            src = torch.cat((lidar_norm, image_norm), dim=0)
            lidar_ffn = self.lidar_linear2(self.lidar_dropout(self.activation(self.lidar_linear1(src[:voxel_num]))))
            image_ffn = self.linear2(self.dropout(self.activation(self.linear1(src[voxel_num:]))))
            src2 = torch.cat((lidar_ffn, image_ffn), dim=0)
            src = src + self.dropout2(src2)
            src = torch.cat((self.lidar_norm2(src[:voxel_num]), self.norm2(src[voxel_num:])), dim=0)
        else:
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
        return src


class _Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, split_ffn):
        super().__init__()
        self.win_attn = _SetAttention(d_model, nhead, dim_feedforward, dropout,
                                      activation=activation, split_ffn=split_ffn)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, set_voxel_inds, set_voxel_masks, pos, voxel_num):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds, voxel_num)
        return self.norm(src + identity)


class UniTRSparseRegionalAttentionBlock(nn.Module):
    """以两种 shift 顺序执行 UniTR sparse regional attention。"""

    def __init__(self, d_model=128, nhead=8, dim_feedforward=256, dropout=0.0,
                 activation="gelu", split_ffn=True, split_residual=True):
        super().__init__()
        self.encoder_list = nn.ModuleList([
            _Encoder(d_model, nhead, dim_feedforward, dropout, activation, split_ffn),
            _Encoder(d_model, nhead, dim_feedforward, dropout, activation, split_ffn),
        ])
        self.split_residual = split_residual
        self.residual_norm = nn.LayerNorm(d_model)
        if split_residual:
            self.lidar_residual_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, src, set_voxel_inds_list, set_voxel_masks_list, pos_embed_list,
                block_id, voxel_num=0, using_checkpoint=False):
        residual = src.clone()
        output = src
        for shift_id, layer in enumerate(self.encoder_list):
            set_id = block_id % 2
            inds = set_voxel_inds_list[shift_id][set_id]
            masks = set_voxel_masks_list[shift_id][set_id]
            pos = pos_embed_list[shift_id]
            if using_checkpoint and self.training:
                output = checkpoint(layer, output, inds, masks, pos, voxel_num)
            else:
                output = layer(output, inds, masks, pos, voxel_num)
        if self.split_residual:
            output = torch.cat((self.lidar_residual_norm(output[:voxel_num] + residual[:voxel_num]),
                                self.residual_norm(output[voxel_num:] + residual[voxel_num:])), dim=0)
        else:
            output = self.residual_norm(output + residual)
        return output

    def load_original_state_dict(self, state_dict, block_id, strict=True):
        """加载 UniTR backbone 或 detector checkpoint 中指定 block 的参数。"""
        model_prefix = "mm_backbone." if any(
            key.startswith("mm_backbone.stage_0.")
            or key.startswith("mm_backbone.residual_norm_stage_0.")
            or key.startswith("mm_backbone.lidar_residual_norm_stage_0.")
            for key in state_dict
        ) else ""
        prefix = f"{model_prefix}stage_0.{block_id}."
        mapped = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        residual_prefix = f"{model_prefix}residual_norm_stage_0.{block_id}."
        mapped.update({"residual_norm." + key[len(residual_prefix):]: value
                       for key, value in state_dict.items() if key.startswith(residual_prefix)})
        lidar_prefix = f"{model_prefix}lidar_residual_norm_stage_0.{block_id}."
        mapped.update({"lidar_residual_norm." + key[len(lidar_prefix):]: value
                       for key, value in state_dict.items() if key.startswith(lidar_prefix)})
        return self.load_state_dict(mapped, strict=strict)
