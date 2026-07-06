# -*- coding: utf-8 -*-
"""
shared_mlp_1d

说明：
    通用 1D 共享 MLP 模块。
    该模块用一组 Conv1d(kernel_size=1) 在序列点维上共享权重执行通道映射，
    适用于点级特征编码、点位图像特征对齐、frustum 内点特征编码和
    Conv1d 序列通道压缩。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class SharedMLP1D(nn.Module):
    """
    1D 共享 MLP 模块。

    Args:
        in_channels: int
            输入通道数。

        channels: Sequence[int]
            每层输出通道数。最后一个值即模块输出通道。

        use_bn: bool
            是否在每个 Conv1d 后添加 BatchNorm1d。

        activation: bool
            是否添加 ReLU 激活。

        activate_last: bool
            最后一层是否添加 BN/ReLU。False 时最后一层只做线性通道映射。

        input_layout: str
            输入布局，支持 "BCL"、"BLC"、"BTMC"。

        output_layout: str
            输出布局，支持 "BCL"、"BLC"、"BTMC"。默认与 input_layout 一致。

    Shape:
        - "BCL": 输入/输出为 (B, C, L)
        - "BLC": 输入/输出为 (B, L, C)
        - "BTMC": 输入/输出为 (B, T, M, C)
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        use_bn: bool = False,
        activation: bool = True,
        activate_last: bool = False,
        input_layout: str = "BCL",
        output_layout: str | None = None,
    ):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels cannot be empty")

        self.input_layout = input_layout.upper()
        self.output_layout = (output_layout or input_layout).upper()
        self.out_channels = int(channels[-1])

        layers: list[nn.Module] = []
        c_in = int(in_channels)
        last_idx = len(channels) - 1
        for i, c_out in enumerate(channels):
            c_out = int(c_out)
            add_tail = i != last_idx or activate_last
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias=not use_bn))
            if add_tail and use_bn:
                layers.append(nn.BatchNorm1d(c_out))
            if add_tail and activation:
                layers.append(nn.ReLU(inplace=True))
            c_in = c_out

        self.net = nn.Sequential(*layers)

    def _to_bcl(self, x: torch.Tensor):
        if self.input_layout == "BCL":
            return x, None
        if self.input_layout == "BLC":
            return x.transpose(1, 2).contiguous(), None
        if self.input_layout == "BTMC":
            if x.dim() != 4:
                raise ValueError("BTMC layout requires input shape (B,T,M,C)")
            B, T, M, C = x.shape
            return x.reshape(B * T, M, C).transpose(1, 2).contiguous(), (B, T, M)
        raise ValueError(f"Unsupported input_layout: {self.input_layout}")

    def _from_bcl(self, x: torch.Tensor, restore_shape):
        if self.output_layout == "BCL":
            return x
        if self.output_layout == "BLC":
            return x.transpose(1, 2).contiguous()
        if self.output_layout == "BTMC":
            if restore_shape is None:
                raise ValueError("BTMC output requires BTMC input layout")
            B, T, M = restore_shape
            return x.transpose(1, 2).contiguous().reshape(B, T, M, self.out_channels)
        raise ValueError(f"Unsupported output_layout: {self.output_layout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor
                输入特征，形状由 input_layout 指定。

        Returns:
            Tensor
                输出特征，形状由 output_layout 指定。
        """
        x_bcl, restore_shape = self._to_bcl(x)
        y = self.net(x_bcl)
        return self._from_bcl(y, restore_shape)
