# -*- coding: utf-8 -*-
"""SparseFusion 专属 block 共用的非融合基础层。"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from common.linear_projection import linear_projection


class ChannelLayerNorm(nn.Module):
    """原 ``network_modules.LayerNorm`` 的 channels-first ``[B,C,H,W]`` 分支。"""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x - mean) / torch.sqrt(variance + self.eps)) + self.bias[:, None, None]


class ConvLN(nn.Module):
    """原 SparseFusion ``Conv2d-ChannelLayerNorm-optional ReLU``。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activate: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                  ChannelLayerNorm(out_channels)]
        if activate:
            layers.append(nn.ReLU())
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class BasicBlock(nn.Module):
    """原 mmdet pytorch-style 两层 3×3 BasicBlock。"""
    expansion = 1

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None if stride == 1 else nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=stride, bias=True), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ProjectionLayerNorm(nn.Module):
    """对 ``[B,C,N]`` 逐 token 执行 Linear 与 LayerNorm。"""

    def __init__(self, hidden_channels: int, input_channels: Optional[int] = None):
        super().__init__()
        self.feat_proj = nn.Linear(input_channels or hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, query):
        projected = linear_projection(
            query.transpose(1, 2), self.feat_proj.weight, self.feat_proj.bias
        )
        return self.norm(projected).transpose(1, 2)


class SparseFusionLinearLayerNorm(nn.Sequential):
    """保持 ``Sequential.0/1`` checkpoint 键并复用公共 Linear 原语。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.Linear(in_channels, out_channels), nn.LayerNorm(out_channels))

    def forward(self, x):
        return self[1](linear_projection(x, self[0].weight, self[0].bias))


class _ConvBNReLU1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.activate(x)


class SparseFusionConvPredictionHeads(nn.Module):
    """原 FFN 多字段 Conv1d prediction heads，输入输出均为 ``[B,C,N]``。"""

    def __init__(self, in_channels: int, heads: Mapping[str, Sequence[int]], head_channels: int = 64):
        super().__init__()
        self.head_names = tuple(heads)
        for name, spec in heads.items():
            out_channels, count = int(spec[0]), int(spec[1])
            use_bn = bool(spec[2]) if len(spec) > 2 else True
            layers, current = [], in_channels
            for _ in range(count - 1):
                layers.append(_ConvBNReLU1d(current, head_channels, use_bn))
                current = head_channels
            layers.append(nn.Conv1d(current, out_channels, 1, bias=True))
            setattr(self, name, nn.Sequential(*layers))

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name)(x) for name in self.head_names}


class SparseFusionLinearPredictionHeads(nn.Module):
    """原 FFNLN 多字段 Linear prediction heads，公开布局为 ``[B,C,N]``。"""

    def __init__(self, in_channels: int, heads: Mapping[str, Sequence[int]], head_channels: int = 64):
        super().__init__()
        self.head_names = tuple(heads)
        for name, spec in heads.items():
            out_channels, count = int(spec[0]), int(spec[1])
            use_norm = bool(spec[2]) if len(spec) > 2 else True
            layers, current = [], in_channels
            for _ in range(count - 1):
                layers.append(nn.Linear(current, head_channels, bias=not use_norm))
                if use_norm:
                    layers.append(nn.LayerNorm(head_channels))
                layers.append(nn.ReLU(inplace=True))
                current = head_channels
            layers.append(nn.Linear(current, out_channels, bias=True))
            setattr(self, name, nn.Sequential(*layers))

    def forward(self, x):
        sequence = x.transpose(1, 2).contiguous()
        return {name: getattr(self, name)(sequence).transpose(1, 2).contiguous()
                for name in self.head_names}


def ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
    """原 MSDeformAttn CUDA kernel 的 bilinear PyTorch 数学 reference。"""
    batch, _, heads, per_head = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    split_sizes = [int(h * w) for h, w in spatial_shapes.tolist()]
    values = value.split(split_sizes, dim=1)
    grids, sampled = 2.0 * sampling_locations - 1.0, []
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        value_level = values[level].flatten(2).transpose(1, 2).reshape(batch * heads, per_head, height, width)
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(value_level, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    output = (torch.stack(sampled, -2).flatten(-2) * weights).sum(-1)
    return output.view(batch, heads * per_head, queries).transpose(1, 2).contiguous()


class SparseFusionMSDeformAttn(nn.Module):
    """SparseFusion 私有 MSDeformAttn；正式维度默认 C128/L4/H8/P4。"""

    def __init__(self, hidden=128, levels=4, heads=8, points=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden 必须能被 heads 整除")
        self.d_model, self.n_levels, self.n_heads, self.n_points = hidden, levels, heads, points
        self.sampling_offsets = nn.Linear(hidden, heads * levels * points * 2)
        self.attention_weights = nn.Linear(hidden, heads * levels * points)
        self.value_proj = nn.Linear(hidden, hidden)
        self.output_proj = nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        theta = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid = torch.stack((theta.cos(), theta.sin()), -1)
        grid = grid / grid.abs().amax(-1, keepdim=True)
        grid = grid.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for point in range(self.n_points):
            grid[:, :, point] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, input_flatten, spatial_shapes,
                level_start_index, input_padding_mask=None):
        del level_start_index
        batch, length, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(batch, length, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(batch, query.shape[1], self.n_heads,
                                                     self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).view(batch, query.shape[1], self.n_heads, -1).softmax(-1)
        weights = weights.view(batch, query.shape[1], self.n_heads, self.n_levels, self.n_points)
        normalizer = torch.stack((spatial_shapes[:, 1], spatial_shapes[:, 0]), -1)
        locations = reference_points[:, :, None, :, None] + offsets / normalizer[None, None, None, :, None]
        return self.output_proj(ms_deform_attn_core_pytorch(value, spatial_shapes, locations, weights))


class SparseFusionDeformableTransformerDecoderLayer(nn.Module):
    """原 deformable decoder：可选 self-attention→MSDA→FFN。"""

    def __init__(self, hidden, heads, levels=4, ffn=256, dropout=0.1,
                 query_position=None, key_position=None, cross_only=False, points=4):
        super().__init__()
        self.cross_only = cross_only
        if not cross_only:
            self.self_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout)
        self.multihead_attn = SparseFusionMSDeformAttn(hidden, levels, heads, points)
        self.linear1, self.linear2 = nn.Linear(hidden, ffn), nn.Linear(ffn, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(hidden), nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.self_posembed, self.cross_posembed = query_position, key_position

    def forward(self, query, key, query_pos, key_pos, reference_points,
                level_start_index, spatial_shapes, query_padding_mask=None,
                input_padding_mask=None):
        qpos = None if self.self_posembed is None else self.self_posembed(query_pos).permute(2, 0, 1)
        kpos = None if self.cross_posembed is None else self.cross_posembed(key_pos).permute(2, 0, 1)
        query_seq, key_seq = query.permute(2, 0, 1), key.permute(2, 0, 1)
        add = lambda x, p: x if p is None else x + p
        if not self.cross_only:
            positioned = add(query_seq, qpos)
            update = self.self_attn(positioned, positioned, positioned, key_padding_mask=query_padding_mask)[0]
            query_seq = self.norm1(query_seq + self.dropout1(update))
        update = self.multihead_attn(add(query_seq, qpos).permute(1, 0, 2), reference_points,
                                     add(key_seq, kpos).permute(1, 0, 2), spatial_shapes,
                                     level_start_index, input_padding_mask).permute(1, 0, 2)
        query_seq = self.norm2(query_seq + self.dropout2(update))
        update = self.linear2(self.dropout(F.relu(self.linear1(query_seq))))
        return self.norm3(query_seq + self.dropout3(update)).permute(1, 2, 0)


def initialize_like_sparsefusion(module: nn.Module):
    """复现 head.init_weights：高维权重 Xavier、MSDA reset、BN momentum=0.1。"""
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    for child in module.modules():
        if isinstance(child, SparseFusionMSDeformAttn):
            child.reset_parameters()
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            child.momentum = 0.1
