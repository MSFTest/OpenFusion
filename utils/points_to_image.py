# -*- coding: utf-8 -*-
"""
points_to_image

说明：
    该文件将 “3D点 -> 2D像素坐标” 投影原语注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.points_to_image(points_3d, proj_matrix, eps)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将任意 3D 点投影到图像平面，生成对应的 2D 像素坐标。
    仅执行齐次坐标投影计算，不包含任何任务相关后处理
    （例如 2D bbox 生成、归一化、裁剪、过滤等）。

输入支持：
    - 点云点 (point cloud)
    - 体素中心 (voxel centers)
    - BEV 网格中心
    - 3D 检测框角点
    - radar 点
    - proposal 关键点

投影公式（严格几何等价）：
    pixels_homo = P @ [x, y, z, 1]^T
    u = pixels_homo[0] / pixels_homo[2]
    v = pixels_homo[1] / pixels_homo[2]

Args:
    points_3d: (..., 3)
        任意 3D 点集，最后一维必须为 (x, y, z)。
        支持任意前缀维度，例如：
            (N, 3)
            (B, N, 3)
            (K, G, 3)

    proj_matrix: (3, 4) 或 (B, 3, 4)
        相机投影矩阵，对应 K[R|t]。
        要求为：源坐标系 → 图像像素坐标系 的完整投影矩阵

        例如：
        CLOCs / KITTI:
            P = P2 @ R0 @ Tr_velo_to_cam
        LoGoNet:
            P = K @ extrinsic[:3, :]
        VirConv:
            P = proj_matrix（原生即为 K[R|t]）

    eps: float
        深度最小值，用于防止除 0。

Returns:
    pixel_coords: (..., 2)
        每个 3D 点在图像平面对应的像素坐标 (u, v)。
        与输入 points_3d 保持相同的前缀维度。
"""

from __future__ import annotations

import torch
from torch.library import Library




# ============================================================
# 2) Python 实现（CompositeExplicitAutograd：用 PyTorch 张量算子实现，自动支持 autograd）
# ============================================================
def _points_to_image_impl(points_3d: torch.Tensor,
                          proj_matrix: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    具体实现：支持
        - points_3d: (...,3)
        - proj_matrix: (3,4) 或 (B,3,4)
    """
    if points_3d.shape[-1] != 3:
        raise ValueError("points_3d last dim must be 3")

    orig_shape = points_3d.shape[:-1]
    device = points_3d.device
    dtype = points_3d.dtype

    P = proj_matrix.to(device=device, dtype=dtype)

    # ---------- batched projection ----------
    if P.dim() == 3:
        # 要求 points_3d 第 0 维是 batch
        if points_3d.dim() < 2:
            raise ValueError("Batched proj_matrix requires points_3d with batch dim")

        if points_3d.shape[0] != P.shape[0]:
            raise ValueError("Batched proj_matrix requires points_3d with same batch dim as proj_matrix")

        B = P.shape[0]
        pts = points_3d.reshape(B, -1, 3)

        ones = torch.ones((B, pts.shape[1], 1), device=device, dtype=dtype)
        pts_h = torch.cat([pts, ones], dim=-1)               # (B, M, 4)

        # (B, M, 4) @ (B, 4, 3) -> (B, M, 3)
        pixels_h = pts_h @ P.transpose(-1, -2)

        depth = pixels_h[..., 2:3].clamp_min(eps)
        pixels = pixels_h[..., :2] / depth                  # (B, M, 2)

        return pixels.reshape(orig_shape + (2,))

    # ---------- single projection ----------
    pts = points_3d.reshape(-1, 3)
    ones = torch.ones((pts.shape[0], 1), device=device, dtype=dtype)
    pts_h = torch.cat([pts, ones], dim=1)                   # (M, 4)

    pixels_h = pts_h @ P.t()                                # (M, 3)
    depth = pixels_h[:, 2:3].clamp_min(eps)
    pixels = pixels_h[:, :2] / depth                        # (M, 2)

    return pixels.reshape(orig_shape + (2,))


# 注册实现：CompositeExplicitAutograd（默认就能反传）
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("points_to_image", _points_to_image_impl)


# ============================================================
# 3) Meta 实现（用于 shape 推导/编译/某些加速场景；等价于你之前的 register_fake）
# ============================================================
def _points_to_image_meta(points_3d: torch.Tensor,
                          proj_matrix: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    # 只返回形状正确的 meta tensor
    return torch.empty(points_3d.shape[:-1] + (2,), device="meta", dtype=points_3d.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("points_to_image", _points_to_image_meta)


# ============================================================
# 4) 可选：提供同名 Python 包装函数（你也可以直接用 torch.ops.fusionrepo.points_to_image）
# ============================================================
def points_to_image(points_3d: torch.Tensor,
                    proj_matrix: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.points_to_image

    注意：
        如果你项目里统一都用 torch.ops.fusionrepo.xxx 调用，则这个函数也可以不用。
    """
    return torch.ops.fusionrepo.points_to_image(points_3d, proj_matrix, float(eps))