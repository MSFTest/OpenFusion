# -*- coding: utf-8 -*-
"""CMT 图像射线坐标编码层。"""

from __future__ import annotations

import torch
from torch import nn


class ImageCoordinateEncoder(nn.Module):
    """CMT ``psi_im``：``Linear(3D,4C)-ReLU-Linear(4C,C)``。

    原源码位置：``cmt_head.py`` 的 ``rv_embedding``、``_rv_pe`` 与
    ``_rv_query_embed``。射线尾部必须为 ``[depth_num,3]``，坐标是归一化当前
    LiDAR xyz。token 与 query 共享本实例；query 按 bool 可见 mask 清零并沿
    camera 维求和，绝不平均。参数 dtype 决定 MLP 计算 dtype，device 跟随模块。
    """

    def __init__(self, depth_num: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.depth_num = int(depth_num)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Sequential(
            nn.Linear(self.depth_num * 3, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

    def encode_tokens(self, ray_coordinates: torch.Tensor) -> torch.Tensor:
        """将 ``[...,D,3]`` 编码为 ``[...,hidden_dim]``。"""
        if ray_coordinates.shape[-2:] != (self.depth_num, 3):
            raise ValueError("ray_coordinates must end with [depth_num,3]")
        flat = ray_coordinates.reshape(*ray_coordinates.shape[:-2], self.depth_num * 3)
        return self.embedding(flat.to(dtype=self.embedding[0].weight.dtype))

    def encode_queries(self, ray_coordinates: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """编码 ``[B,V,N,D,3]`` 并按 ``visible[B,V,N]`` 沿 V 求和。"""
        if ray_coordinates.ndim != 5 or ray_coordinates.shape[-2:] != (self.depth_num, 3):
            raise ValueError("query rays must have shape [B,V,N,depth_num,3]")
        if visible.shape != ray_coordinates.shape[:3]:
            raise ValueError("visible must have shape [B,V,N]")
        embeddings = self.encode_tokens(ray_coordinates)
        return (embeddings * visible.unsqueeze(-1).to(embeddings.dtype)).sum(dim=1)

    def forward(self, ray_coordinates: torch.Tensor, visible: torch.Tensor = None) -> torch.Tensor:
        """无 mask 时编码 token；有 mask 时返回多视角求和后的 query 编码。"""
        if visible is None:
            return self.encode_tokens(ray_coordinates)
        return self.encode_queries(ray_coordinates, visible)

    @staticmethod
    def original_state_dict_key_map() -> dict:
        """返回原 CmtHead 到当前层的无损权重键映射。"""
        return {
            "rv_embedding.0.weight": "embedding.0.weight",
            "rv_embedding.0.bias": "embedding.0.bias",
            "rv_embedding.2.weight": "embedding.2.weight",
            "rv_embedding.2.bias": "embedding.2.bias",
        }
