# -*- coding: utf-8 -*-
"""SparseFusion 九维 view position 描述构造算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def _impl(lidar_geometry: torch.Tensor, normal_image_position: torch.Tensor) -> torch.Tensor:
    if lidar_geometry.ndim != 3 or lidar_geometry.shape[-1] != 10:
        raise ValueError("lidar_geometry 必须是 [B,P,10]")
    if normal_image_position.shape != lidar_geometry.shape[:2] + (2,):
        raise ValueError("normal_image_position 必须是 [B,P,2]")
    position = lidar_geometry[..., :7].clone()
    position[..., :2] = _inverse_sigmoid((position[..., :2] + 12.0) / 204.0)
    position[..., 2] = _inverse_sigmoid((position[..., 2] + 10.0) / 20.0)
    position[..., 3:5] = _inverse_sigmoid((position[..., 3:5] + 1.0) / 2.0)
    return torch.cat((position, normal_image_position), -1)


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("sparsefusion_view_position_descriptor", _impl)


def _meta(lidar_geometry: torch.Tensor, normal_image_position: torch.Tensor) -> torch.Tensor:
    if lidar_geometry.ndim != 3 or lidar_geometry.shape[-1] != 10:
        raise ValueError("lidar_geometry 必须是 [B,P,10]")
    if normal_image_position.shape != lidar_geometry.shape[:2] + (2,):
        raise ValueError("normal_image_position 必须是 [B,P,2]")
    dtype = torch.promote_types(lidar_geometry.dtype, normal_image_position.dtype)
    return torch.empty((*lidar_geometry.shape[:2], 9), device="meta", dtype=dtype)


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("sparsefusion_view_position_descriptor", _meta)


def sparsefusion_view_position_descriptor(lidar_geometry: torch.Tensor,
                                          normal_image_position: torch.Tensor) -> torch.Tensor:
    """由 LiDAR 几何 ``[B,P,10]`` 和图像位置 logit ``[B,P,2]`` 生成
    ``[B,P,9]`` 描述。输入输出 dtype/device 一致；LiDAR xy 为 BEV 网格单位，
    z/dim/vel为原预测单位，normal image position 不再次 sigmoid。算子只使用
    xyz、rot2、vel2，按原固定范围归一化后 inverse-sigmoid，dim 不参与。
    """
    return torch.ops.fusionrepo.sparsefusion_view_position_descriptor(
        lidar_geometry, normal_image_position
    )
