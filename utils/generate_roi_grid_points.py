# -*- coding: utf-8 -*-
"""
generate_roi_grid_points

说明：
    该文件实现 “ROI -> 3D网格采样点” 通用原语，
    供 operator_registry.py 中注册后的
    torch.ops.fusionrepo.generate_roi_grid_points 调用。

功能：
    在每个 3D ROI 内部生成规则三维采样网格。

    仅执行几何采样计算，不包含任何任务相关操作
    （例如特征采样、ROI pooling、anchor生成等）。

输入：
    rois: (K, 7)
        3D ROI 参数，格式为：
        (x, y, z, dx, dy, dz, yaw)

    grid_size: int
        每个维度上的网格划分数量

输出：
    grid_points: (K, G, 3)
        K = ROI 数量
        G = grid_size^3
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 1) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _generate_roi_grid_points_impl(
    rois: torch.Tensor,
    grid_size: int
) -> torch.Tensor:
    if rois.dim() != 2 or rois.shape[1] != 7:
        raise ValueError("rois must have shape (K, 7)")

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    device = rois.device
    dtype = rois.dtype

    K = rois.shape[0]
    G = grid_size ** 3

    lin = torch.linspace(-0.5, 0.5, grid_size, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    base_grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (G, 3)

    grid_points = torch.zeros((K, G, 3), device=device, dtype=dtype)

    for i in range(K):
        x, y, z, dx, dy, dz, yaw = rois[i]

        pts = base_grid.clone()
        pts[:, 0] *= dx
        pts[:, 1] *= dy
        pts[:, 2] *= dz

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        rot = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, torch.zeros((), device=device, dtype=dtype)]),
                torch.stack([sin_yaw,  cos_yaw, torch.zeros((), device=device, dtype=dtype)]),
                torch.stack([torch.zeros((), device=device, dtype=dtype),
                             torch.zeros((), device=device, dtype=dtype),
                             torch.ones((), device=device, dtype=dtype)])
            ],
            dim=0
        )  # (3, 3)

        pts = pts @ rot.t()

        pts[:, 0] += x
        pts[:, 1] += y
        pts[:, 2] += z

        grid_points[i] = pts

    return grid_points


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("generate_roi_grid_points", _generate_roi_grid_points_impl)


# ============================================================
# 2) Meta 实现
# ============================================================
def _generate_roi_grid_points_meta(
    rois: torch.Tensor,
    grid_size: int
) -> torch.Tensor:
    K = rois.shape[0]
    G = grid_size ** 3
    return torch.empty((K, G, 3), device="meta", dtype=rois.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("generate_roi_grid_points", _generate_roi_grid_points_meta)


# ============================================================
# 3) Python 包装
# ============================================================
def generate_roi_grid_points(
    rois: torch.Tensor,
    grid_size: int
) -> torch.Tensor:
    return torch.ops.fusionrepo.generate_roi_grid_points(rois, int(grid_size))