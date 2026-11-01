# -*- coding: utf-8 -*-
"""CMT 分类与回归 heads 共享的分组预测基础层。"""

from __future__ import annotations

import torch
from torch import nn


class GroupLayerNorm1d(nn.Module):
    """按 decoder-layer group 独立统计通道均值与方差。

    原源码位置：``cmt_head.py:LayerNormFunction/GroupLayerNorm1d``。输入输出均为
    ``[B,C,L]``；每组在自身 ``C/groups`` 通道上归一化，每个通道保留独立
    可训练 weight/bias，方差使用 biased 统计，默认 ``eps=1e-6``。
    """

    def __init__(self, channels: int, groups: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.groups = int(groups)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化 contiguous 或 non-contiguous ``[B,C,L]`` 张量。"""
        if x.ndim != 3 or x.shape[1] != self.weight.numel():
            raise ValueError("x must have shape [B,channels,L]")
        batch, channels, length = x.shape
        grouped = x.reshape(batch, self.groups, channels // self.groups, length)
        mean = grouped.mean(dim=2, keepdim=True)
        variance = (grouped - mean).pow(2).mean(dim=2, keepdim=True)
        normalized = ((grouped - mean) / torch.sqrt(variance + self.eps)).reshape(
            batch, channels, length
        )
        return self.weight.view(1, channels, 1) * normalized + self.bias.view(1, channels, 1)


class GroupedPredictionBranch(nn.Sequential):
    """每个 decoder 层独立参数的两层 1x1 Conv1d 分支。

    原源码位置：``cmt_head.py:SeparateTaskHead``。结构精确为 grouped
    ``Conv1d(C,64,bias=False)-GroupLayerNorm1d-ReLU(inplace)-Conv1d(64,O)``；
    两个卷积都使用 Kaiming fan_out 初始化，末层 bias 初始为零（分类 head
    随后覆盖为 -2.19）。输入 ``[B,groups*C,N]``，输出 ``[B,groups*O,N]``。
    """

    def __init__(
        self,
        output_channels: int,
        in_channels: int = 256,
        head_channels: int = 64,
        groups: int = 6,
    ) -> None:
        super().__init__(
            nn.Conv1d(in_channels * groups, head_channels * groups, 1, groups=groups, bias=False),
            GroupLayerNorm1d(head_channels * groups, groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_channels * groups, output_channels * groups, 1, groups=groups, bias=True),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def stacked_queries_to_grouped(queries: torch.Tensor, groups: int) -> torch.Tensor:
    """原 ``n b q c -> b (n c) q``，输入 ``[groups,B,N,C]``。"""
    if queries.ndim != 4 or queries.shape[0] != groups:
        raise ValueError("queries must have shape [groups,B,N,C]")
    return queries.permute(1, 0, 3, 2).reshape(
        queries.shape[1], groups * queries.shape[3], queries.shape[2]
    )


def grouped_to_stacked(output: torch.Tensor, groups: int, channels: int) -> torch.Tensor:
    """原 ``b (n c) q -> n b q c``，返回 ``[groups,B,N,channels]``。"""
    if output.ndim != 3 or output.shape[1] != groups * channels:
        raise ValueError("output must have shape [B,groups*channels,N]")
    batch, _, queries = output.shape
    return output.reshape(batch, groups, channels, queries).permute(1, 0, 3, 2)
