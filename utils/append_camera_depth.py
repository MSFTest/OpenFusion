# -*- coding: utf-8 -*-
"""
append_camera_depth

说明：
    该文件将“补充相机深度并拼接为 (u, v, d)”注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.append_camera_depth(points_3d, extrinsic, pixel_coords, eps)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    将源坐标系中的 3D 点通过外参矩阵变换到相机坐标系，
    取相机坐标系 z 作为深度 d，
    再与已有的图像平面坐标 pixel_coords=(u,v) 拼接，
    生成 frustum / image-depth 坐标 (u,v,d)。

输入支持：
    - 单样本：
        points_3d:    (. , 3)
        extrinsic:    (4, 4) 或 (3, 4)
        pixel_coords: (. , 2)
    - Batch：
        points_3d:    (B, . , 3)
        extrinsic:    (B, 4, 4) 或 (B, 3, 4)
        pixel_coords: (B, . , 2)

Args:
    points_3d: (. , 3)
        源坐标系中的 3D 点，最后一维必须为 (x, y, z)。

    extrinsic: (4, 4) / (3, 4) 或其 batch 形式
        源坐标系到相机坐标系的外参矩阵。

    pixel_coords: (. , 2)
        已计算好的图像平面像素坐标 (u, v)，其前缀维度必须与 points_3d 一致。

    eps: float
        深度最小值，用于防止数值不稳定；会对相机坐标系 z 做 clamp_min。

Returns:
    frustum_coords: (. , 3)
        拼接后的 (u, v, d)，其中 d 为相机坐标系深度。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _append_camera_depth_impl(
    points_3d: torch.Tensor,
    extrinsic: torch.Tensor,
    pixel_coords: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    具体实现：支持
        - points_3d: (. ,3)
        - extrinsic: (4,4)/(3,4) 或 (B,4,4)/(B,3,4)
        - pixel_coords: (. ,2)
    """
    if points_3d.shape[-1] != 3:
        raise ValueError("points_3d last dim must be 3")
    if pixel_coords.shape[-1] != 2:
        raise ValueError("pixel_coords last dim must be 2")
    if points_3d.shape[:-1] != pixel_coords.shape[:-1]:
        raise ValueError("points_3d and pixel_coords must share the same prefix shape")
    if extrinsic.dim() not in (2, 3):
        raise ValueError("extrinsic must have shape (4,4)/(3,4) or batched version")
    if extrinsic.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError("extrinsic last two dims must be (4,4) or (3,4)")

    orig_shape = points_3d.shape[:-1]
    device = points_3d.device
    dtype = points_3d.dtype

    pts = points_3d.to(device=device, dtype=dtype)
    uv = pixel_coords.to(device=device, dtype=dtype)
    T = extrinsic.to(device=device, dtype=dtype)

    # ---------- batched extrinsic ----------
    if T.dim() == 3:
        if pts.dim() < 2:
            raise ValueError("Batched extrinsic requires points_3d with batch dim")
        if pts.shape[0] != T.shape[0]:
            raise ValueError("Batched extrinsic requires same batch size as points_3d")
        if uv.shape[0] != T.shape[0]:
            raise ValueError("Batched extrinsic requires same batch size as pixel_coords")

        B = T.shape[0]
        pts_flat = pts.reshape(B, -1, 3)
        uv_flat = uv.reshape(B, -1, 2)

        ones = torch.ones((B, pts_flat.shape[1], 1), device=device, dtype=dtype)
        pts_h = torch.cat([pts_flat, ones], dim=-1)  # (B, M, 4)

        if T.shape[-2:] == (4, 4):
            cam_pts_h = pts_h @ T.transpose(-1, -2)   # (B, M, 4)
            depth = cam_pts_h[..., 2:3].clamp_min(eps)
        else:
            cam_pts = pts_h @ T.transpose(-1, -2)     # (B, M, 3)
            depth = cam_pts[..., 2:3].clamp_min(eps)

        frustum = torch.cat([uv_flat, depth], dim=-1)  # (B, M, 3)
        return frustum.reshape(orig_shape + (3,))

    # ---------- single extrinsic ----------
    pts_flat = pts.reshape(-1, 3)
    uv_flat = uv.reshape(-1, 2)

    ones = torch.ones((pts_flat.shape[0], 1), device=device, dtype=dtype)
    pts_h = torch.cat([pts_flat, ones], dim=-1)  # (M, 4)

    if T.shape[-2:] == (4, 4):
        cam_pts_h = pts_h @ T.t()                 # (M, 4)
        depth = cam_pts_h[:, 2:3].clamp_min(eps)
    else:
        cam_pts = pts_h @ T.t()                   # (M, 3)
        depth = cam_pts[:, 2:3].clamp_min(eps)

    frustum = torch.cat([uv_flat, depth], dim=-1)  # (M, 3)
    return frustum.reshape(orig_shape + (3,))


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("append_camera_depth", _append_camera_depth_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _append_camera_depth_meta(
    points_3d: torch.Tensor,
    extrinsic: torch.Tensor,
    pixel_coords: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    return torch.empty(
        points_3d.shape[:-1] + (3,),
        device="meta",
        dtype=points_3d.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("append_camera_depth", _append_camera_depth_meta)


# ============================================================
# 4) 可选 Python 包装
# ============================================================
def append_camera_depth(
    points_3d: torch.Tensor,
    extrinsic: torch.Tensor,
    pixel_coords: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.append_camera_depth
    """
    return torch.ops.fusionrepo.append_camera_depth(
        points_3d, extrinsic, pixel_coords, float(eps)
    )