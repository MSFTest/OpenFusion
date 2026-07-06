# -*- coding: utf-8 -*-
"""
epnet_lidar_point_extractor

说明：
    EPNeT 私有规范化算子，注册为：
        torch.ops.fusionrepo.epnet_lidar_point_extractor(pointcloud)

    从原始点云特征中提取前三个 xyz 坐标通道，保留 EPNeT 原始输入输出语义。

Args:
    pointcloud: Tensor
        原始点云，shape 为 (B, N, C)，C >= 3。

Returns:
    Tensor
        xyz 坐标，shape 为 (B, N, 3)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnet_lidar_point_extractor_impl(pointcloud: torch.Tensor) -> torch.Tensor:
    if pointcloud.dim() != 3 or pointcloud.shape[-1] < 3:
        raise ValueError("pointcloud must have shape (B, N, C>=3)")
    return pointcloud[..., 0:3].contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnet_lidar_point_extractor", _epnet_lidar_point_extractor_impl)


def _epnet_lidar_point_extractor_meta(pointcloud: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (*pointcloud.shape[:-1], 3),
        device="meta",
        dtype=pointcloud.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnet_lidar_point_extractor", _epnet_lidar_point_extractor_meta)


def epnet_lidar_point_extractor(pointcloud: torch.Tensor) -> torch.Tensor:
    return torch.ops.fusionrepo.epnet_lidar_point_extractor(pointcloud)
