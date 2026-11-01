# -*- coding: utf-8 -*-
"""CMT 投影 query 射线的固定深度三维坐标构造算子。"""

from __future__ import annotations

import torch
from torch.library import Library

from utils.batched_homogeneous_transform_and_normalize import (
    batched_homogeneous_transform_and_normalize,
)


def _impl(projected_rays, img2lidar, pc_range, depth_num: int):
    """执行原 CmtHead._rv_query_embed 的射线采样和反投影。"""
    if projected_rays.ndim != 4 or projected_rays.shape[-1] != 3:
        raise ValueError("projected_rays must have shape [B,V,N,3]")
    batch, views, queries = projected_rays.shape[:3]
    if img2lidar.shape != (batch, views, 4, 4):
        raise ValueError("img2lidar must have shape [B,V,4,4]")
    if pc_range.numel() != 6 or depth_num <= 0:
        raise ValueError("pc_range must have 6 values and depth_num must be positive")
    pc = pc_range.detach().to(device=projected_rays.device, dtype=projected_rays.dtype)
    depths = 1.0 + torch.arange(
        depth_num, device=projected_rays.device, dtype=projected_rays.dtype
    ) * (pc[3] - 1.0) / depth_num
    points = (projected_rays[..., None, :] * depths[:, None]).reshape(
        batch * views, queries * depth_num, 3
    )
    transforms = img2lidar.detach().to(
        device=projected_rays.device, dtype=projected_rays.dtype
    ).reshape(batch * views, 4, 4)
    lidar_points = batched_homogeneous_transform_and_normalize(
        points, transforms, False, 1e-6
    ).reshape(batch, views, queries, depth_num, 3)
    return (lidar_points - pc[:3]) / (pc[3:] - pc[:3])


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("cmt_projected_ray_coordinates", _impl)


def _meta(projected_rays, img2lidar, pc_range, depth_num: int):
    if projected_rays.ndim != 4 or projected_rays.shape[-1] != 3:
        raise ValueError("projected_rays must have shape [B,V,N,3]")
    batch, views = projected_rays.shape[:2]
    if img2lidar.shape != (batch, views, 4, 4):
        raise ValueError("img2lidar must have shape [B,V,4,4]")
    if pc_range.numel() != 6 or depth_num <= 0:
        raise ValueError("pc_range must have 6 values and depth_num must be positive")
    return torch.empty(
        (*projected_rays.shape[:-1], depth_num, 3),
        device="meta",
        dtype=projected_rays.dtype,
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("cmt_projected_ray_coordinates", _meta)


def cmt_projected_ray_coordinates(
    projected_rays: torch.Tensor,
    img2lidar: torch.Tensor,
    pc_range: torch.Tensor,
    depth_num: int,
) -> torch.Tensor:
    """沿投影 anchor 射线构造固定深度点并反投影到 LiDAR 坐标。

    Args:
        projected_rays: ``[B,V,N,3]``，透视除法后的 ``(u,v,约1)``，允许
            非 contiguous，连续坐标路径可微。
        img2lidar: ``[B,V,4,4]``，列向量约定下增强图像坐标到当前 LiDAR
            米制坐标的齐次变换，按原 metadata 语义 detach。
        pc_range: ``[xmin,ymin,zmin,xmax,ymax,zmax]``，单位米。
        depth_num: 每条射线的离散深度数 ``D>0``。

    Returns:
        ``[B,V,N,D,3]``，与 projected_rays 同 dtype/device、contiguous；深度为
        ``1+k*(xmax-1)/D``，输出按 pc_range 归一化但不 clamp、不按可见性删除。
        空 ``N`` 返回精确空 shape；标定矩阵与 pc_range 不求导。
    """
    return torch.ops.fusionrepo.cmt_projected_ray_coordinates(
        projected_rays, img2lidar, pc_range, int(depth_num)
    )
