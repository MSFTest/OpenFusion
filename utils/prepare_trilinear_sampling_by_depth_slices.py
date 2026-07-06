# -*- coding: utf-8 -*-
"""
prepare_trilinear_sampling_by_depth_slices

说明：
    该文件将“3D Frustum 三线性采样的前置准备”注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.prepare_trilinear_sampling_by_depth_slices(
            frustum_features,
            frustum_coords,
            depth_bins,
            height,
            width
        )

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将 HMFI 中的 3D Frustum 采样问题拆解为：
        1. 两个深度切片上的 2D 双线性采样
        2. 一次深度方向线性融合

    本算子完成第 1 步之前的准备工作：
        - 从 frustum_coords 中取出 (u, v, d)
        - 由 d 计算连续深度索引 r
        - 取相邻深度层 r0, r1
        - 为每个采样点构造对应的 2D feature map 切片
        - 构造 grid_sample 所需的归一化 2D grid
        - 计算深度方向融合权重 depth_alpha

输入：
    frustum_features : (H, W, R, C)
    frustum_coords   : (N, 3)   其中每行为 (u, v, d)
    depth_bins       : (H, W, R)
    height           : int
    width            : int

输出：
    List[Tensor] :
        - feature_map0 : (N, C, H, W)
        - feature_map1 : (N, C, H, W)
        - grid         : (N, 1, 2)   坐标范围 [-1, 1]
        - depth_alpha  : (N, 1)
"""

from __future__ import annotations

from typing import List

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现
# ============================================================
def _prepare_trilinear_sampling_by_depth_slices_impl(
    frustum_features: torch.Tensor,
    frustum_coords: torch.Tensor,
    depth_bins: torch.Tensor,
    height: int,
    width: int
) -> List[torch.Tensor]:
    """
    具体实现：
        frustum_features : (H, W, R, C)
        frustum_coords   : (N, 3)
        depth_bins       : (H, W, R)

    返回：
        [
            feature_map0,   # (N, C, H, W)
            feature_map1,   # (N, C, H, W)
            grid,           # (N, 1, 2)
            depth_alpha     # (N, 1)
        ]
    """
    if frustum_features.dim() != 4:
        raise ValueError("frustum_features must have shape (H, W, R, C)")

    if frustum_coords.dim() != 2 or frustum_coords.shape[1] != 3:
        raise ValueError("frustum_coords must have shape (N, 3)")

    if depth_bins.dim() != 3:
        raise ValueError("depth_bins must have shape (H, W, R)")

    H, W, R, C = frustum_features.shape
    Hb, Wb, Rb = depth_bins.shape

    if (H, W, R) != (Hb, Wb, Rb):
        raise ValueError("depth_bins shape must match frustum_features spatial/depth dims")

    if height != H or width != W:
        raise ValueError("height/width must match frustum_features shape")

    device = frustum_features.device
    dtype = frustum_features.dtype
    N = frustum_coords.shape[0]

    # --------------------------------------------------------
    # 1. 拆坐标
    # --------------------------------------------------------
    u = frustum_coords[:, 0].to(device=device, dtype=dtype).clamp(0, width - 1)
    v = frustum_coords[:, 1].to(device=device, dtype=dtype).clamp(0, height - 1)
    d = frustum_coords[:, 2].to(device=device, dtype=dtype)

    # --------------------------------------------------------
    # 2. 深度 -> 连续深度索引 r
    #    与 HMFI 原实现保持一致：使用全局 d_min / d_max
    # --------------------------------------------------------
    d_min = depth_bins[..., 0].min().to(device=device, dtype=dtype)
    d_max = depth_bins[..., -1].max().to(device=device, dtype=dtype)

    r = (d - d_min) / (d_max - d_min + 1e-6) * (R - 1)
    r = r.clamp(0, R - 1)

    r0 = torch.floor(r).long()
    r1 = torch.clamp(r0 + 1, max=R - 1)
    depth_alpha = (r - r0.to(dtype)).unsqueeze(-1)  # (N,1)

    # --------------------------------------------------------
    # 3. 构造两个深度切片 feature map
    #    每个采样点单独对应一对切片：
    #        feature_map0[i] = frustum_features[:, :, r0_i, :]
    #        feature_map1[i] = frustum_features[:, :, r1_i, :]
    #
    #    结果形状：
    #        (N, C, H, W)
    # --------------------------------------------------------
    frustum_perm = frustum_features.permute(2, 3, 0, 1).contiguous()  # (R, C, H, W)

    feature_map0 = frustum_perm.index_select(0, r0)  # (N, C, H, W)
    feature_map1 = frustum_perm.index_select(0, r1)  # (N, C, H, W)

    # --------------------------------------------------------
    # 4. 构造 2D grid，归一化到 [-1, 1]
    #    feature_sampling_by_grid 要求：
    #        grid : (B, N, 2) 或 (B, N, 1, 2)
    #
    #    这里每个样本只采一个点，因此：
    #        grid : (N, 1, 2)
    # --------------------------------------------------------
    if width > 1:
        u_norm = 2.0 * u / (width - 1) - 1.0
    else:
        u_norm = torch.zeros_like(u)

    if height > 1:
        v_norm = 2.0 * v / (height - 1) - 1.0
    else:
        v_norm = torch.zeros_like(v)

    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1)  # (N,1,2)

    return [
        feature_map0.contiguous(),
        feature_map1.contiguous(),
        grid.contiguous(),
        depth_alpha.contiguous(),
    ]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "prepare_trilinear_sampling_by_depth_slices",
    _prepare_trilinear_sampling_by_depth_slices_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================
def _prepare_trilinear_sampling_by_depth_slices_meta(
    frustum_features: torch.Tensor,
    frustum_coords: torch.Tensor,
    depth_bins: torch.Tensor,
    height: int,
    width: int
) -> List[torch.Tensor]:
    H, W, _, C = frustum_features.shape
    N = frustum_coords.shape[0]

    feature_map0 = torch.empty((N, C, H, W), device="meta", dtype=frustum_features.dtype)
    feature_map1 = torch.empty((N, C, H, W), device="meta", dtype=frustum_features.dtype)
    grid = torch.empty((N, 1, 2), device="meta", dtype=frustum_coords.dtype)
    depth_alpha = torch.empty((N, 1), device="meta", dtype=frustum_coords.dtype)

    return [feature_map0, feature_map1, grid, depth_alpha]


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "prepare_trilinear_sampling_by_depth_slices",
    _prepare_trilinear_sampling_by_depth_slices_meta
)


# ============================================================
# 4) Python 包装
# ============================================================
def prepare_trilinear_sampling_by_depth_slices(
    frustum_features: torch.Tensor,
    frustum_coords: torch.Tensor,
    depth_bins: torch.Tensor,
    height: int,
    width: int
) -> List[torch.Tensor]:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.prepare_trilinear_sampling_by_depth_slices
    """
    return torch.ops.fusionrepo.prepare_trilinear_sampling_by_depth_slices(
        frustum_features,
        frustum_coords,
        depth_bins,
        int(height),
        int(width)
    )