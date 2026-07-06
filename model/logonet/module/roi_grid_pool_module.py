# -*- coding: utf-8 -*-
"""
roi_grid_pool_module

说明：
    该模块使用已经提取并注册的通用函数算子，组合实现 LoGoNet 中
    ROI 网格生成 + ROI 内点云特征收集 + 池化 这一阶段逻辑。

模块内部调用的通用算子：
    - torch.ops.fusionrepo.generate_roi_grid_points
    - torch.ops.fusionrepo.gather_roi_features
    - torch.ops.fusionrepo.reduce_pool

功能流程：
    rois
        ↓
    generate_roi_grid_points
        ↓
    grid_points

    rois + point_coords + point_features
        ↓
    gather_roi_features
        ↓
    gathered_features, valid_counts

    gathered_features
        ↓
    reduce_pool
        ↓
    pooled_features

同时补充输出：
    - roi_centers
    - grid_point_counts

输入：
    rois: (K, 7)
        ROI 参数，格式为：
        (x, y, z, dx, dy, dz, yaw)

    point_coords: (N, 3)
        点 / voxel 的三维坐标

    point_features: (N, C)
        点 / voxel 的特征

输出：
    {
        "grid_points": (K, G, 3),
        "roi_centers": (K, 3),
        "grid_point_counts": (K, G),
        "gathered_features": (K, N, C),
        "valid_counts": (K,),
        "pooled_features": (K, C)
    }
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class RoIGridPoolModule(nn.Module):
    """
    ROI 网格池化模块：
        ROI
          ↓
        生成网格点
          ↓
        收集 ROI 内点云特征
          ↓
        池化得到 ROI 特征
    """

    def __init__(
        self,
        grid_size: int = 6,
        pool_mode: str = "mean"
    ):
        super().__init__()

        if pool_mode not in ("max", "sum", "mean"):
            raise ValueError("pool_mode must be one of: 'max', 'sum', 'mean'")

        self.grid_size = int(grid_size)
        self.pool_mode = str(pool_mode)

    def _compute_grid_point_counts(
        self,
        grid_points: torch.Tensor,
        point_coords: torch.Tensor,
        rois: torch.Tensor
    ) -> torch.Tensor:
        """
        统计每个 ROI 的每个网格点附近分到的点数量。

        这里采用最简单且与后续位置编码兼容的方式：
            1. 先把点变到 ROI 局部坐标系
            2. 按 grid_size 在 ROI 内做规则分箱
            3. 统计每个网格单元里的点数

        输入：
            grid_points: (K, G, 3)   这里只用来取 K、G
            point_coords: (N, 3)
            rois: (K, 7)

        输出：
            grid_point_counts: (K, G)
        """
        device = point_coords.device
        dtype = point_coords.dtype

        K, G, _ = grid_points.shape
        N = point_coords.shape[0]

        grid_point_counts = torch.zeros(
            (K, G),
            device=device,
            dtype=dtype
        )

        for i in range(K):
            cx, cy, cz, dx, dy, dz, yaw = rois[i]

            # ------------------------------------------------
            # 1) 点移到 ROI 局部坐标系
            # ------------------------------------------------
            rel = point_coords - rois[i, 0:3].unsqueeze(0)   # (N, 3)

            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)

            rot_inv = torch.stack(
                [
                    torch.stack([ cos_yaw, sin_yaw, torch.zeros((), device=device, dtype=dtype)]),
                    torch.stack([-sin_yaw, cos_yaw, torch.zeros((), device=device, dtype=dtype)]),
                    torch.stack([ torch.zeros((), device=device, dtype=dtype),
                                  torch.zeros((), device=device, dtype=dtype),
                                  torch.ones((), device=device, dtype=dtype)])
                ],
                dim=0
            )  # (3, 3)

            rel_local = rel @ rot_inv.t()  # (N, 3)

            # ------------------------------------------------
            # 2) 只保留 ROI 内部点
            # ------------------------------------------------
            half_dx = dx / 2.0
            half_dy = dy / 2.0
            half_dz = dz / 2.0

            inside_mask = (
                (rel_local[:, 0] >= -half_dx) & (rel_local[:, 0] <= half_dx) &
                (rel_local[:, 1] >= -half_dy) & (rel_local[:, 1] <= half_dy) &
                (rel_local[:, 2] >= -half_dz) & (rel_local[:, 2] <= half_dz)
            )

            if not inside_mask.any():
                continue

            pts_in = rel_local[inside_mask]  # (Ni, 3)

            # ------------------------------------------------
            # 3) 映射到 [0, 1]
            # ------------------------------------------------
            eps = 1e-6
            norm_x = (pts_in[:, 0] + half_dx) / max(float(dx.item()), eps)
            norm_y = (pts_in[:, 1] + half_dy) / max(float(dy.item()), eps)
            norm_z = (pts_in[:, 2] + half_dz) / max(float(dz.item()), eps)

            norm_x = torch.clamp(norm_x, 0.0, 1.0 - 1e-6)
            norm_y = torch.clamp(norm_y, 0.0, 1.0 - 1e-6)
            norm_z = torch.clamp(norm_z, 0.0, 1.0 - 1e-6)

            # ------------------------------------------------
            # 4) 分箱到 grid_size^3 个网格
            # ------------------------------------------------
            gx = torch.floor(norm_x * self.grid_size).long()
            gy = torch.floor(norm_y * self.grid_size).long()
            gz = torch.floor(norm_z * self.grid_size).long()

            flat_idx = (
                gx * (self.grid_size * self.grid_size) +
                gy * self.grid_size +
                gz
            )  # (Ni,)

            counts = torch.bincount(
                flat_idx,
                minlength=G
            ).to(dtype)

            grid_point_counts[i] = counts

        return grid_point_counts

    def forward(
        self,
        rois: torch.Tensor,
        point_coords: torch.Tensor,
        point_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rois: (K, 7)
            point_coords: (N, 3)
            point_features: (N, C)

        Returns:
            {
                "grid_points": (K, G, 3),
                "roi_centers": (K, 3),
                "grid_point_counts": (K, G),
                "gathered_features": (K, N, C),
                "valid_counts": (K,),
                "pooled_features": (K, C)
            }
        """
        if rois.dim() != 2 or rois.shape[1] != 7:
            raise ValueError("rois must have shape (K, 7)")

        if point_coords.dim() != 2 or point_coords.shape[1] != 3:
            raise ValueError("point_coords must have shape (N, 3)")

        if point_features.dim() != 2:
            raise ValueError("point_features must have shape (N, C)")

        if point_coords.shape[0] != point_features.shape[0]:
            raise ValueError("point_coords and point_features must have same first dim")

        # ------------------------------------------------
        # 1) 生成 ROI 网格点
        # ------------------------------------------------
        grid_points = torch.ops.fusionrepo.generate_roi_grid_points(
            rois,
            int(self.grid_size)
        )  # (K, G, 3)

        # ------------------------------------------------
        # 2) ROI 中心
        # ------------------------------------------------
        roi_centers = rois[:, 0:3]  # (K, 3)

        # ------------------------------------------------
        # 3) 统计每个网格的点数
        # ------------------------------------------------
        grid_point_counts = self._compute_grid_point_counts(
            grid_points=grid_points,
            point_coords=point_coords,
            rois=rois
        )  # (K, G)

        # ------------------------------------------------
        # 4) 收集 ROI 内特征
        # ------------------------------------------------
        gathered_features, valid_counts = torch.ops.fusionrepo.gather_roi_features(
            rois,
            point_coords,
            point_features
        )  # (K, N, C), (K,)

        # ------------------------------------------------
        # 5) 池化
        # ------------------------------------------------
        pooled_features = torch.ops.fusionrepo.reduce_pool(
            gathered_features,
            1,
            self.pool_mode,
            False
        )  # (K, C)

        return {
            "grid_points": grid_points,
            "roi_centers": roi_centers,
            "grid_point_counts": grid_point_counts,
            "gathered_features": gathered_features,
            "valid_counts": valid_counts,
            "pooled_features": pooled_features
        }