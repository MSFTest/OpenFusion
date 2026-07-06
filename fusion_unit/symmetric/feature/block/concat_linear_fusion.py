# -*- coding: utf-8 -*-
"""
concat_linear_fusion

说明：
    该文件实现“拼接后线性融合”层形式通用算子，
    用于在多路特征完成拼接后，进一步进行线性映射、可选归一化与激活。

功能：
    先调用已注册的函数形式通用算子：
        torch.ops.fusionrepo.feature_concatenation(features, dim)

    再执行：
        linear projection -> optional layernorm -> activation

    该模块适用于 HMFI 中 QFM_FeatureFusion_Concat 这类
    “拼接不是终点，拼接后还要继续做融合映射”的情况。

结构：
    feature_concatenation
        -> Linear
        -> optional LayerNorm
        -> optional Activation

Args:
    in_channels: int
        拼接后的总输入通道数。

    out_channels: int
        输出通道数。

    concat_dim: int, optional
        特征拼接维度，默认 -1。

    bias: bool, optional
        线性层是否使用偏置，默认 False。

    use_layernorm: bool, optional
        是否在 linear 后使用 LayerNorm，默认 False。

    activation: str, optional
        激活函数类型，可选：
            - "relu"
            - "gelu"
            - "none"
        默认 "relu"。

Returns:
    out: torch.Tensor
        融合后的输出特征。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConcatLinearFusion(nn.Module):
    """
    拼接后线性融合模块。

    用于将多路特征先拼接，再映射到统一特征空间，
    并可选执行归一化与激活。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        concat_dim: int = -1,
        bias: bool = False,
        use_layernorm: bool = False,
        activation: str = "relu"
    ):
        super().__init__()

        self.concat_dim = concat_dim
        self.use_layernorm = use_layernorm
        self.activation_type = activation.lower()

        self.fusion_proj = nn.Linear(
            in_channels,
            out_channels,
            bias=bias
        )

        if self.use_layernorm:
            self.norm = nn.LayerNorm(out_channels)

        if self.activation_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.activation_type == "gelu":
            self.activation = nn.GELU()
        elif self.activation_type == "none":
            self.activation = None
        else:
            raise ValueError(
                f"Unsupported activation type: {activation}. "
                f"Expected one of ['relu', 'gelu', 'none']."
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播。

        Args:
            features: List[torch.Tensor]
                待融合的多路特征列表。
                所有张量除拼接维外，其余维度必须一致。

        Returns:
            out: torch.Tensor
                融合后的输出特征。
        """
        if not isinstance(features, (list, tuple)):
            raise TypeError("features must be a list or tuple of torch.Tensor")

        if len(features) == 0:
            raise ValueError("features cannot be empty")

        fused = torch.ops.fusionrepo.feature_concatenation(
            list(features),
            int(self.concat_dim)
        )

        out = self.fusion_proj(fused)

        if self.use_layernorm:
            out = self.norm(out)

        if self.activation is not None:
            out = self.activation(out)

        return out