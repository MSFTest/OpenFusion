# -*- coding: utf-8 -*-
"""DeepInteraction 无筛选 bbox 解码算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _impl(center: torch.Tensor, height: torch.Tensor, dimension: torch.Tensor,
          rotation: torch.Tensor, velocity: torch.Tensor,
          point_cloud_range: torch.Tensor, voxel_size: torch.Tensor,
          out_size_factor: int) -> torch.Tensor:
    """将 query 回归字段解码为 LiDAR bottom-center box。

    Args:
        center: ``[B,2,K]``，BEV 特征格坐标 ``(x,y)``。
        height: ``[B,1,K]``，米制 box 几何中心 z。
        dimension: ``[B,3,K]``，``log(dx,dy,dz)``。
        rotation: ``[B,2,K]``，``(sin(yaw),cos(yaw))``。
        velocity: ``[B,2,K]``，LiDAR 坐标系米/秒速度。
        point_cloud_range: ``[6]``，米制 ``(xmin,ymin,zmin,xmax,ymax,zmax)``。
        voxel_size: ``[3]``，米制 voxel 尺寸。
        out_size_factor: BEV 相对 voxel 网格的整数步长。
    Returns:
        ``[B,K,9]``，dtype/device 跟随输入，字段为
        ``(x,y,bottom_z,dx,dy,dz,yaw,vx,vy)``。不做范围筛选。

    原源码：``transfusion_bbox_coder.py:39-126``。
    """
    point_cloud_range = point_cloud_range.to(device=center.device, dtype=center.dtype)
    voxel_size = voxel_size.to(device=center.device, dtype=center.dtype)
    x = center[:, 0] * out_size_factor * voxel_size[0] + point_cloud_range[0]
    y = center[:, 1] * out_size_factor * voxel_size[1] + point_cloud_range[1]
    size = dimension.exp().permute(0, 2, 1)
    bottom_z = height[:, 0] - size[..., 2] * 0.5
    yaw = torch.atan2(rotation[:, 0], rotation[:, 1])
    return torch.cat((x[..., None], y[..., None], bottom_z[..., None], size,
                      yaw[..., None], velocity.permute(0, 2, 1)), dim=-1)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("deepinteraction_decode_boxes", _impl)


def _meta(center, height, dimension, rotation, velocity, point_cloud_range,
          voxel_size, out_size_factor: int):
    return center.new_empty((center.shape[0], center.shape[-1], 9))


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("deepinteraction_decode_boxes", _meta)


def deepinteraction_decode_boxes(center, height, dimension, rotation, velocity,
                                 point_cloud_range, voxel_size,
                                 out_size_factor: int):
    """调用 ``torch.ops.fusionrepo.deepinteraction_decode_boxes``。"""
    return torch.ops.fusionrepo.deepinteraction_decode_boxes(
        center, height, dimension, rotation, velocity, point_cloud_range,
        voxel_size, out_size_factor
    )
