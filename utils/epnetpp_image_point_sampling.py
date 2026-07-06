# -*- coding: utf-8 -*-
"""
epnetpp_image_point_sampling

说明：
    EPNet++ 点级图像特征采样补全算子，注册为：
        torch.ops.fusionrepo.epnetpp_image_point_sampling(Fi, xy)

    该算子保留 EPNet++ 原始 image_point_sampling 的输入输出语义，内部复用
    通用 feature_sampling_by_grid 完成双线性采样。

Args:
    Fi: Tensor
        图像特征图，shape 为 (B, C, H, W)。
    xy: Tensor
        点在图像特征图上的归一化坐标，shape 为 (B, N, 2)，坐标范围为
        [-1, 1]，顺序为 (x, y)，与 torch.nn.functional.grid_sample 一致。

Returns:
    Tensor
        点级图像特征，shape 为 (B, C, N)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _epnetpp_image_point_sampling_impl(
    Fi: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.feature_sampling_by_grid(Fi, xy).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("epnetpp_image_point_sampling", _epnetpp_image_point_sampling_impl)


def _epnetpp_image_point_sampling_meta(
    Fi: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    batch_size, channels, _, _ = Fi.shape
    num_points = xy.shape[1]
    return torch.empty((batch_size, channels, num_points), device="meta", dtype=Fi.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("epnetpp_image_point_sampling", _epnetpp_image_point_sampling_meta)


def epnetpp_image_point_sampling(
    Fi: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.epnetpp_image_point_sampling(Fi, xy)
