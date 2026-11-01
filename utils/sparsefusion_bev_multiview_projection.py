# -*- coding: utf-8 -*-
"""SparseFusion BEV 柱中心到多相机投影算子。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library

from utils.bevdet_build_reference_points import build_reference_points


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def _impl(bev_pos, min_height, max_height, lidar2img, valid_shapes,
          image_height: int, image_width: int, out_size_factor: float,
          voxel_x: float, voxel_y: float, pc_min_x: float, pc_min_y: float,
          image_stride: float) -> Tuple[torch.Tensor, ...]:
    if bev_pos.ndim != 3 or bev_pos.shape[-1] != 2:
        raise ValueError("bev_pos 必须是 [B,N,2]")
    batch, count = bev_pos.shape[:2]
    if min_height.shape != max_height.shape or min_height.numel() != batch * count:
        raise ValueError("min/max_height 必须对应每个 BEV 格")
    if lidar2img.ndim != 4 or lidar2img.shape[0] != batch or lidar2img.shape[-2:] != (4, 4):
        raise ValueError("lidar2img 必须是 [B,V,4,4]")
    if valid_shapes.shape != lidar2img.shape[:2] + (2,):
        raise ValueError("valid_shapes 必须是 [B,V,2]")
    height = ((min_height.reshape(batch, count) + max_height.reshape(batch, count)) / 2.0)
    base_factor = int(out_size_factor)
    if base_factor == 0:
        base_factor = 1
    metric_xyz = build_reference_points(
        bev_pos.transpose(1, 2), height[:, None],
        bev_pos.new_tensor((voxel_x, voxel_y)), bev_pos.new_tensor((pc_min_x, pc_min_y)),
        base_factor,
    )
    # 原接口允许 float out_size_factor；补回非整数倍率，正式配置为整数 8。
    if float(out_size_factor) != base_factor:
        scale = float(out_size_factor) / base_factor
        metric_xyz = metric_xyz.clone()
        metric_xyz[..., 0] = (metric_xyz[..., 0] - pc_min_x) * scale + pc_min_x
        metric_xyz[..., 1] = (metric_xyz[..., 1] - pc_min_y) * scale + pc_min_y
    points = torch.cat((metric_xyz, torch.ones_like(metric_xyz[..., :1])), -1)
    projected = torch.matmul(points[:, None], lidar2img.transpose(-1, -2))
    depth = projected[..., 2].clamp(min=1e-5)
    xy = projected[..., :2] / depth[..., None] / image_stride
    valid_w, valid_h = valid_shapes[..., 0] / image_stride, valid_shapes[..., 1] / image_stride
    visible = ((xy[..., 0] >= 0) & (xy[..., 0] < valid_w[..., None]) &
               (xy[..., 1] >= 0) & (xy[..., 1] < valid_h[..., None]) &
               (height > -50)[:, None])
    reference = xy / xy.new_tensor((image_width, image_height))
    query_pos = torch.cat((_inverse_sigmoid(reference), depth.log().unsqueeze(-1)), -1)
    return reference, query_pos, visible, depth, metric_xyz


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("sparsefusion_bev_multiview_projection", _impl)


def _meta(bev_pos, min_height, max_height, lidar2img, valid_shapes,
          image_height: int, image_width: int, out_size_factor: float,
          voxel_x: float, voxel_y: float, pc_min_x: float, pc_min_y: float,
          image_stride: float):
    if bev_pos.ndim != 3 or bev_pos.shape[-1] != 2:
        raise ValueError("bev_pos 必须是 [B,N,2]")
    batch, count = bev_pos.shape[:2]
    if min_height.shape != max_height.shape or min_height.numel() != batch * count:
        raise ValueError("min/max_height 必须对应每个 BEV 格")
    if lidar2img.ndim != 4 or lidar2img.shape[0] != batch or lidar2img.shape[-2:] != (4, 4):
        raise ValueError("lidar2img 必须是 [B,V,4,4]")
    if valid_shapes.shape != lidar2img.shape[:2] + (2,):
        raise ValueError("valid_shapes 必须是 [B,V,2]")
    height_dtype = torch.promote_types(min_height.dtype, max_height.dtype)
    output_dtype = torch.promote_types(bev_pos.dtype, height_dtype)
    if output_dtype != lidar2img.dtype:
        raise RuntimeError("metric_xyz 与 lidar2img 的 dtype 必须一致")
    del image_height, image_width
    del out_size_factor, voxel_x, voxel_y, pc_min_x, pc_min_y, image_stride
    views = lidar2img.shape[1]
    opts = {"device": "meta", "dtype": output_dtype}
    return (torch.empty((batch, views, count, 2), **opts),
            torch.empty((batch, views, count, 3), **opts),
            torch.empty((batch, views, count), device="meta", dtype=torch.bool),
            torch.empty((batch, views, count), **opts), torch.empty((batch, count, 3), **opts))


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("sparsefusion_bev_multiview_projection", _meta)


def sparsefusion_bev_multiview_projection(*args) -> Tuple[torch.Tensor, ...]:
    """将 BEV 网格中心投影到全部相机。

    输入 BEV 坐标为 ``[B,N,2]`` 网格单位，高度为米，``lidar2img`` 是列向量
    约定的增强 LiDAR 米制坐标到图像齐次矩阵 ``[B,V,4,4]``；valid shape
    末维为像素 ``(width,height)``。输出 reference ``[B,V,N,2]``、位置描述
    ``[B,V,N,3]``、可见 mask、截断后深度和米制 xyz。深度按原实现下限
    ``1e-5``，不额外以正深度剔除；浮点 dtype/device 跟随 ``bev_pos``。
    """
    return torch.ops.fusionrepo.sparsefusion_bev_multiview_projection(*args)
