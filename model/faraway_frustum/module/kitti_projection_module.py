# -*- coding: utf-8 -*-
"""KITTI 标定投影模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _to_4x4(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = matrix.new_zeros((4, 4))
        out[:3, :4] = matrix
        out[3, 3] = 1.0
        return out
    if matrix.shape == (3, 3):
        out = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
        out[:3, :3] = matrix
        return out
    raise ValueError(f"Unsupported calibration matrix shape: {tuple(matrix.shape)}")


class KittiProjectionModule(nn.Module):
    """执行 KITTI LiDAR 到 camera rect，再到图像平面的投影。

    坐标系说明：
        输入点云为 KITTI Velodyne/LiDAR 坐标 ``(x,y,z,intensity)``；
        ``Tr_velo_to_cam`` 将其变换到 camera 坐标，``R0_rect`` 做 rect 校正，
        ``P2`` 将 rect camera 坐标投影到 image_2 像素坐标。

    Inputs:
        points_lidar: ``(N, 3+)``，LiDAR 坐标点云。
        calib: dict，包含 ``P2 (3,4)``、``R0_rect (3,3)``、
            ``Tr_velo_to_cam (3,4)``。

    Returns:
        dict，包含 ``points_3d_cam0 (M,3)``、``points_2d_img (M,2)``、
        ``valid_mask (N,)``。
    """

    def forward(self, points_lidar: torch.Tensor, calib: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = points_lidar.device
        dtype = points_lidar.dtype
        p2 = calib["P2"].to(device=device, dtype=dtype)
        r0 = calib["R0_rect"].to(device=device, dtype=dtype)
        tr = calib["Tr_velo_to_cam"].to(device=device, dtype=dtype)
        r0_4x4 = _to_4x4(r0)
        tr_4x4 = _to_4x4(tr)

        xyz = points_lidar[:, :3]
        ones = torch.ones((xyz.shape[0], 1), dtype=dtype, device=device)
        lidar_h = torch.cat([xyz, ones], dim=1)
        cam_h = (r0_4x4 @ tr_4x4 @ lidar_h.t()).t()
        points_cam = cam_h[:, :3]
        valid = points_cam[:, 2] > 1e-3
        points_cam_valid = points_cam[valid]

        cam_h_valid = torch.cat(
            [points_cam_valid, torch.ones((points_cam_valid.shape[0], 1), dtype=dtype, device=device)],
            dim=1,
        )
        projected = (p2 @ cam_h_valid.t()).t()
        uv = projected[:, :2] / projected[:, 2:3].clamp(min=1e-6)
        return {
            "points_3d_cam0": points_cam_valid,
            "points_2d_img": uv,
            "valid_mask": valid,
        }
