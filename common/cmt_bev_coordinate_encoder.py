# -*- coding: utf-8 -*-
"""CMT 点云/BEV 坐标编码层。"""

from __future__ import annotations

import math

import torch
from torch import nn


def pos2embed(pos: torch.Tensor, num_pos_feats: int = 256) -> torch.Tensor:
    """将归一化 ``(x,y)`` 编码为 CMT 使用的正余弦位置向量。

    Args:
        pos: ``[...,K]``，``K>=2``；仅使用当前 LiDAR-BEV 平面的归一化 x、y。
        num_pos_feats: 每个坐标的编码宽度，必须为正偶数。

    Returns:
        ``[...,2*num_pos_feats]``，同 device、float32 计算布局，顺序为 y 后 x。
    """
    if pos.shape[-1] < 2:
        raise ValueError("pos last dimension must contain x and y")
    if num_pos_feats <= 0 or num_pos_feats % 2:
        raise ValueError("num_pos_feats must be a positive even integer")
    scaled = pos * (2.0 * math.pi)
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = scaled[..., 0, None] / dim_t
    pos_y = scaled[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), -1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), -1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class BEVCoordinateEncoder(nn.Module):
    """CMT ``psi_pc``：``Linear(2C,C)-ReLU-Linear(C,C)``。

    原源码位置：``CMT-master/.../dense_heads/cmt_head.py`` 的 ``pos2embed`` 与
    ``CmtHead.bev_embedding``。输入 ``[...,2]`` 为归一化 LiDAR-BEV ``(x,y)``；
    输出 ``[...,hidden_dim]``。模块含两层可训练线性参数，ReLU 为 inplace，
    同一实例应由 BEV token 与 query 路径共享。
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """编码 ``[...,2]`` 坐标，返回同 device 的 ``[...,hidden_dim]``。"""
        encoded = pos2embed(coordinates, self.hidden_dim)
        return self.embedding(encoded.to(dtype=self.embedding[0].weight.dtype))

    @staticmethod
    def original_state_dict_key_map() -> dict:
        """返回原 CmtHead 到当前层的无损权重键映射。"""
        return {
            "bev_embedding.0.weight": "embedding.0.weight",
            "bev_embedding.0.bias": "embedding.0.bias",
            "bev_embedding.2.weight": "embedding.2.weight",
            "bev_embedding.2.bias": "embedding.2.bias",
        }
