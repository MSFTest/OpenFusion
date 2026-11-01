# -*- coding: utf-8 -*-
"""SparseFusion 相机 query 检测属性到 LiDAR 几何的转换算子。"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library


def _impl(center_2d, depth_2d, dim_2d, rot_2d, vel_2d, view_index,
          lidar2cam, cam_intrinsic, image_height: int, image_width: int,
          pc_min_x: float, pc_min_y: float, pc_max_x: float, pc_max_y: float,
          bev_width: int, bev_height: int) -> Tuple[torch.Tensor, ...]:
    batch, proposals = view_index.shape
    if center_2d.shape != (batch, 2, proposals):
        raise ValueError("center_2d 必须是 [B,2,P]")
    if any(x.ndim != 3 for x in (depth_2d, dim_2d, rot_2d, vel_2d)):
        raise ValueError("预测字段必须是 [B,C,P]")
    if lidar2cam.ndim != 4 or cam_intrinsic.shape != lidar2cam.shape:
        raise ValueError("标定必须是同 shape 的 [B,V,4,4]")
    batch_ids = torch.arange(batch, device=view_index.device)[:, None].expand(-1, proposals)
    selected_l2c, selected_k = lidar2cam[batch_ids, view_index], cam_intrinsic[batch_ids, view_index]
    center_live, depth_live = center_2d.transpose(1, 2), depth_2d.transpose(1, 2)[..., :1]
    pixels = center_live * center_live.new_tensor((image_width, image_height))
    camera_h = torch.cat((pixels * depth_live, depth_live), -1)
    loc_cam_live = torch.matmul(torch.linalg.inv(selected_k[..., :3, :3]), camera_h.unsqueeze(-1)).squeeze(-1)
    loc_cam = loc_cam_live.detach().clone()
    rotation, translation = selected_l2c[..., :3, :3], selected_l2c[..., :3, 3]
    inverse_rotation = torch.linalg.inv(rotation)
    lidar_xyz = torch.matmul(inverse_rotation, (loc_cam - translation).unsqueeze(-1)).squeeze(-1)
    lidar_xyz[..., 0] = (lidar_xyz[..., 0] - pc_min_x) / (pc_max_x - pc_min_x) * bev_width
    lidar_xyz[..., 1] = (lidar_xyz[..., 1] - pc_min_y) / (pc_max_y - pc_min_y) * bev_height
    dims = dim_2d.detach().clone()[:, [2, 0, 1]].transpose(1, 2)
    camera_rot = torch.cat((rot_2d[:, 1:2].detach(), torch.zeros_like(rot_2d[:, :1]),
                            -rot_2d[:, :1].detach()), 1).transpose(1, 2)
    lidar_rot = -torch.matmul(inverse_rotation, camera_rot.unsqueeze(-1)).squeeze(-1)[..., [0, 1]]
    camera_vel = torch.cat((vel_2d[:, 0:1].detach(), torch.zeros_like(vel_2d[:, :1]),
                            vel_2d[:, 1:2].detach()), 1).transpose(1, 2)
    lidar_vel = torch.matmul(inverse_rotation, camera_vel.unsqueeze(-1)).squeeze(-1)[..., [0, 1]]
    geometry = torch.cat((lidar_xyz, lidar_rot, lidar_vel, dims), -1)
    camera_info = torch.cat((rotation.reshape(batch, proposals, 9), translation,
                             selected_k[..., 0, 0, None], selected_k[..., 1, 1, None],
                             selected_k[..., :2, 2]), -1).transpose(1, 2).contiguous()
    return geometry, camera_info, loc_cam_live.transpose(1, 2).contiguous()


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("sparsefusion_camera_query_to_lidar", _impl)


def _meta(center_2d, depth_2d, dim_2d, rot_2d, vel_2d, view_index,
          lidar2cam, cam_intrinsic, image_height: int, image_width: int,
          pc_min_x: float, pc_min_y: float, pc_max_x: float, pc_max_y: float,
          bev_width: int, bev_height: int):
    if view_index.ndim != 2:
        raise ValueError("view_index 必须是 [B,P]")
    batch, proposals = view_index.shape
    if center_2d.shape != (batch, 2, proposals):
        raise ValueError("center_2d 必须是 [B,2,P]")
    field_contracts = (
        (depth_2d, 1, "depth_2d"), (dim_2d, 3, "dim_2d"),
        (rot_2d, 2, "rot_2d"), (vel_2d, 2, "vel_2d"),
    )
    for tensor, channels, name in field_contracts:
        if (tensor.ndim != 3 or tensor.shape[0] != batch or
                tensor.shape[1] < channels or tensor.shape[2] != proposals):
            raise ValueError(f"{name} 的 shape 不符合 [B,C,P] 契约")
    if (lidar2cam.ndim != 4 or lidar2cam.shape[0] != batch or
            lidar2cam.shape[-2:] != (4, 4) or cam_intrinsic.shape != lidar2cam.shape):
        raise ValueError("标定必须是同 shape 的 [B,V,4,4]")

    camera_dtype = torch.promote_types(center_2d.dtype, depth_2d.dtype)
    if camera_dtype != cam_intrinsic.dtype:
        raise RuntimeError("camera_h 与 cam_intrinsic 的 dtype 必须一致")
    lidar_dtype = lidar2cam.dtype
    if torch.promote_types(camera_dtype, lidar_dtype) != lidar_dtype:
        raise RuntimeError("camera location 与 lidar2cam 的 dtype 不兼容")
    if rot_2d.dtype != lidar_dtype or vel_2d.dtype != lidar_dtype:
        raise RuntimeError("rot/vel 与 lidar2cam 的 dtype 必须一致")

    del image_height, image_width, pc_min_x, pc_min_y, pc_max_x, pc_max_y, bev_width, bev_height
    geometry_dtype = torch.promote_types(lidar_dtype, dim_2d.dtype)
    camera_info_dtype = torch.promote_types(lidar_dtype, cam_intrinsic.dtype)
    return (
        torch.empty((batch, proposals, 10), device="meta", dtype=geometry_dtype),
        torch.empty((batch, 16, proposals), device="meta", dtype=camera_info_dtype),
        torch.empty((batch, 3, proposals), device="meta", dtype=camera_dtype),
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("sparsefusion_camera_query_to_lidar", _meta)


def sparsefusion_camera_query_to_lidar(*args) -> Tuple[torch.Tensor, ...]:
    """把相机坐标预测转换为 LiDAR-view query 几何。

    center/depth/dim/rot/vel 为 ``[B,C,P]``，center 位于增强图像 ``[0,1]``，
    depth/dim/速度单位为米制，view index 为 ``long[B,P]``。标定为
    ``[B,V,4,4]`` 的 LiDAR→camera 与 camera intrinsic。返回 LiDAR BEV
    网格几何 ``[B,P,10]``、camera info ``[B,16,P]`` 和可微 camera location
    ``[B,3,P]``。进入 LiDAR 几何分支的 camera location及框属性按原源码
    detach；重复相机索引保持 query 顺序，奇异矩阵直接报错。
    """
    return torch.ops.fusionrepo.sparsefusion_camera_query_to_lidar(*args)
