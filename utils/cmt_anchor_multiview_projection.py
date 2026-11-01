# -*- coding: utf-8 -*-
"""CMT 归一化三维 anchor 到多相机的投影与可见性算子。"""

from __future__ import annotations

import torch
from torch.library import Library

from utils.batched_homogeneous_transform_and_normalize import (
    batched_homogeneous_transform_and_normalize,
)


def _impl(reference_points, lidar2img, pc_range, pad_h: int, pad_w: int):
    """执行原 CmtHead._rv_query_embed 的 anchor 投影。"""
    if reference_points.ndim != 3 or reference_points.shape[-1] != 3:
        raise ValueError("reference_points must have shape [B,N,3]")
    if lidar2img.ndim != 4 or lidar2img.shape[-2:] != (4, 4):
        raise ValueError("lidar2img must have shape [B,V,4,4]")
    batch, queries = reference_points.shape[:2]
    if lidar2img.shape[0] != batch:
        raise ValueError("reference_points and lidar2img batch dimensions must match")
    if pc_range.numel() != 6:
        raise ValueError("pc_range must have 6 values")
    views = lidar2img.shape[1]
    pc = pc_range.detach().to(device=reference_points.device, dtype=reference_points.dtype)
    metric = reference_points * (pc[3:] - pc[:3]) + pc[:3]
    points = metric[:, None].expand(batch, views, queries, 3).reshape(batch * views, queries, 3)
    transforms = lidar2img.detach().to(
        device=reference_points.device, dtype=reference_points.dtype
    ).reshape(batch * views, 4, 4)
    projected = batched_homogeneous_transform_and_normalize(
        points, transforms, False, 1e-6
    ).reshape(batch, views, queries, 3)
    projected_clone = projected.clone()
    positive_depth = projected_clone[..., 2:3].detach() > 0
    denominator = (
        projected[..., 2:3].detach()
        + positive_depth.to(projected.dtype) * 1e-6
        - (~positive_depth).to(projected.dtype) * 1e-6
    )
    projected_clone[..., :3] = projected[..., :3] / denominator
    visible = (
        (projected_clone[..., 0] >= 0)
        & (projected_clone[..., 0] < float(pad_w))
        & (projected_clone[..., 1] >= 0)
        & (projected_clone[..., 1] < float(pad_h))
        & positive_depth.squeeze(-1)
    )
    return reference_points[..., :2], projected_clone[..., :3], visible


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("cmt_anchor_multiview_projection", _impl)


def _meta(reference_points, lidar2img, pc_range, pad_h: int, pad_w: int):
    del pad_h, pad_w
    if reference_points.ndim != 3 or reference_points.shape[-1] != 3:
        raise ValueError("reference_points must have shape [B,N,3]")
    if lidar2img.ndim != 4 or lidar2img.shape[-2:] != (4, 4):
        raise ValueError("lidar2img must have shape [B,V,4,4]")
    batch, queries = reference_points.shape[:2]
    if lidar2img.shape[0] != batch:
        raise ValueError("reference_points and lidar2img batch dimensions must match")
    if pc_range.numel() != 6:
        raise ValueError("pc_range must have 6 values")
    views = lidar2img.shape[1]
    return (
        torch.empty((batch, queries, 2), device="meta", dtype=reference_points.dtype),
        torch.empty((batch, views, queries, 3), device="meta", dtype=reference_points.dtype),
        torch.empty((batch, views, queries), device="meta", dtype=torch.bool),
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("cmt_anchor_multiview_projection", _meta)


def cmt_anchor_multiview_projection(
    reference_points: torch.Tensor,
    lidar2img: torch.Tensor,
    pc_range: torch.Tensor,
    pad_h: int,
    pad_w: int,
):
    """将 CMT anchor 投到 BEV 与全部相机并生成可见性。

    Args:
        reference_points: ``[B,N,3]``，当前 LiDAR 范围内归一化 ``xyz``。
        lidar2img: ``[B,V,4,4]``，列向量约定下当前 LiDAR 米制坐标到增强图像
            齐次坐标的矩阵；按原 metadata 语义 detach。
        pc_range: ``[xmin,ymin,zmin,xmax,ymax,zmax]``，单位米。
        pad_h: 图像有效高度，边界采用 ``0<=v<pad_h``。
        pad_w: 图像有效宽度，边界采用 ``0<=u<pad_w``。

    Returns:
        三元组：``bev[B,N,2]``、``rays[B,V,N,3]``、
        ``visible bool[B,V,N]``；dtype/device 继承 reference_points。
        深度分母 detach，正/非正深度分别加/减 ``1e-6``，不 clamp 负深度；
        分子路径可微，可见性离散判断不可微。
    """
    return torch.ops.fusionrepo.cmt_anchor_multiview_projection(
        reference_points, lidar2img, pc_range, int(pad_h), int(pad_w)
    )
