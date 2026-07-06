# -*- coding: utf-8 -*-
"""
image_voxel_downsampling

说明：
    该文件将“图像体素特征下采样”注册为 PyTorch 自定义算子：

        torch.ops.fusionrepo.image_voxel_downsampling(
            image_voxel_features,
            downsample_factor,
            mode
        )

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    对图像体素特征进行下采样，兼容两种输入形态：
        1. 稠密网格体素： (X, Y, Z, C)
        2. 列表体素特征： (N, C)

模式编码：
    - 0：Max
    - 1：Avg

说明：
    - 对稠密体素网格，采用 3D pooling 下采样
    - 对列表体素特征，采用 stride 子采样
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library


# ============================================================
# 2) Python 实现（CompositeExplicitAutograd）
# ============================================================
def _image_voxel_downsampling_impl(
    image_voxel_features: torch.Tensor,
    downsample_factor: int,
    mode: int
) -> torch.Tensor:
    """
    具体实现：
        - 稠密网格输入： (X, Y, Z, C)
        - 列表输入：     (N, C)
    """
    if downsample_factor <= 1:
        return image_voxel_features.contiguous()

    # =========================================================
    # Case A: 稠密体素网格 (X, Y, Z, C)
    # =========================================================
    if image_voxel_features.dim() == 4:
        voxel_5d = image_voxel_features.permute(3, 2, 1, 0).unsqueeze(0)  # (1, C, Z, Y, X)

        if mode == 0:
            pooled = F.max_pool3d(
                voxel_5d,
                kernel_size=downsample_factor,
                stride=downsample_factor
            )
        elif mode == 1:
            pooled = F.avg_pool3d(
                voxel_5d,
                kernel_size=downsample_factor,
                stride=downsample_factor
            )
        else:
            raise ValueError(f"Unsupported downsampling mode code: {mode}")

        return pooled.squeeze(0).permute(3, 2, 1, 0).contiguous()  # (X', Y', Z', C)

    # =========================================================
    # Case B: 列表体素特征 (N, C)
    # =========================================================
    if image_voxel_features.dim() == 2:
        return image_voxel_features[::downsample_factor].clone().contiguous()

    raise ValueError(
        f"Unsupported image_voxel_features.dim(): {image_voxel_features.dim()}, expected 2 or 4."
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl(
    "image_voxel_downsampling",
    _image_voxel_downsampling_impl
)


# ============================================================
# 3) Meta 实现
# ============================================================
def _image_voxel_downsampling_meta(
    image_voxel_features: torch.Tensor,
    downsample_factor: int,
    mode: int
) -> torch.Tensor:
    if downsample_factor <= 1:
        return torch.empty_like(image_voxel_features, device="meta")

    if image_voxel_features.dim() == 4:
        x, y, z, c = image_voxel_features.shape
        f = max(int(downsample_factor), 1)

        x_out = (x - f) // f + 1 if x >= f else 0
        y_out = (y - f) // f + 1 if y >= f else 0
        z_out = (z - f) // f + 1 if z >= f else 0

        x_out = max(x_out, 1)
        y_out = max(y_out, 1)
        z_out = max(z_out, 1)

        return torch.empty((x_out, y_out, z_out, c), device="meta", dtype=image_voxel_features.dtype)

    if image_voxel_features.dim() == 2:
        n, c = image_voxel_features.shape
        f = max(int(downsample_factor), 1)
        n_out = (n + f - 1) // f
        return torch.empty((n_out, c), device="meta", dtype=image_voxel_features.dtype)

    raise ValueError(
        f"Unsupported image_voxel_features.dim(): {image_voxel_features.dim()}, expected 2 or 4."
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl(
    "image_voxel_downsampling",
    _image_voxel_downsampling_meta
)


# ============================================================
# 4) Python 包装
# ============================================================
def image_voxel_downsampling(
    image_voxel_features: torch.Tensor,
    downsample_factor: int,
    mode: int
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.image_voxel_downsampling
    """
    return torch.ops.fusionrepo.image_voxel_downsampling(
        image_voxel_features,
        int(downsample_factor),
        int(mode)
    )