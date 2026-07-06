# -*- coding: utf-8 -*-
"""
build_projection_matrix

说明：
    该文件将“相机内参 + 外参 -> 完整图像投影矩阵”注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.build_projection_matrix(cam_intrinsic, extrinsic)

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    根据相机内参矩阵 K 与外参矩阵 [R|t] / T，
    构造源坐标系到图像像素坐标系的完整投影矩阵 P = K[R|t]。

输入支持：
    - 单样本：
        cam_intrinsic: (3, 3)
        extrinsic:     (4, 4) 或 (3, 4)
    - Batch：
        cam_intrinsic: (B, 3, 3)
        extrinsic:     (B, 4, 4) 或 (B, 3, 4)

典型用途：
    - HMFI:
        P = K @ lidar_to_cam[:3, :]
    - LoGoNet:
        P = K @ extrinsic[:3, :]
    - 一般多模态投影任务：
        源坐标系 3D 点 -> 图像像素坐标

Args:
    cam_intrinsic: (3, 3) 或 (B, 3, 3)
        相机内参矩阵 K。

    extrinsic: (4, 4) / (3, 4) 或其 batch 形式
        源坐标系到相机坐标系的外参矩阵。
        若输入为 (4, 4)，内部自动取前 3 行构造 [R|t]。

Returns:
    proj_matrix: (3, 4) 或 (B, 3, 4)
        完整图像投影矩阵 P = K[R|t]。
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _build_projection_matrix_impl(
    cam_intrinsic: torch.Tensor,
    extrinsic: torch.Tensor
) -> torch.Tensor:
    """
    具体实现：支持
        - cam_intrinsic: (3,3) 或 (B,3,3)
        - extrinsic: (4,4)/(3,4) 或 (B,4,4)/(B,3,4)
    """
    if cam_intrinsic.dim() not in (2, 3):
        raise ValueError("cam_intrinsic must have shape (3,3) or (B,3,3)")
    if extrinsic.dim() not in (2, 3):
        raise ValueError("extrinsic must have shape (4,4)/(3,4) or batched version")

    if cam_intrinsic.shape[-2:] != (3, 3):
        raise ValueError("cam_intrinsic last two dims must be (3,3)")
    if extrinsic.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError("extrinsic last two dims must be (4,4) or (3,4)")

    device = cam_intrinsic.device
    dtype = cam_intrinsic.dtype

    K = cam_intrinsic.to(device=device, dtype=dtype)
    T = extrinsic.to(device=device, dtype=dtype)

    if T.shape[-2:] == (4, 4):
        T = T[..., :3, :]  # (..., 3, 4)

    if K.dim() != T.dim():
        raise ValueError("cam_intrinsic and extrinsic must both be single or both be batched")

    if K.dim() == 3 and K.shape[0] != T.shape[0]:
        raise ValueError("Batched cam_intrinsic and extrinsic must have same batch size")

    # 单样本: (3,3) @ (3,4) -> (3,4)
    # Batch : (B,3,3) @ (B,3,4) -> (B,3,4)
    proj_matrix = K @ T
    return proj_matrix


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("build_projection_matrix", _build_projection_matrix_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _build_projection_matrix_meta(
    cam_intrinsic: torch.Tensor,
    extrinsic: torch.Tensor
) -> torch.Tensor:
    if cam_intrinsic.dim() == 3:
        return torch.empty(
            (cam_intrinsic.shape[0], 3, 4),
            device="meta",
            dtype=cam_intrinsic.dtype
        )
    return torch.empty((3, 4), device="meta", dtype=cam_intrinsic.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("build_projection_matrix", _build_projection_matrix_meta)


# ============================================================
# 4) 可选 Python 包装
# ============================================================
def build_projection_matrix(
    cam_intrinsic: torch.Tensor,
    extrinsic: torch.Tensor
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.build_projection_matrix
    """
    return torch.ops.fusionrepo.build_projection_matrix(cam_intrinsic, extrinsic)