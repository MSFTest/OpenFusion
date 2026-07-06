# -*- coding: utf-8 -*-
"""
generate_3d_anchor_grid

输出：
    anchors_3d: (Nx * Nz * M, 7)
        每行为 [x, y, z, dx, dy, dz, yaw]
"""

from __future__ import annotations

import torch
from torch.library import Library


def _generate_3d_anchor_grid_impl(
    x_centers: torch.Tensor,
    z_centers: torch.Tensor,
    y_center: float,
    anchor_sizes: torch.Tensor
) -> torch.Tensor:
    if x_centers.dim() != 1:
        raise ValueError("x_centers must have shape (Nx,)")
    if z_centers.dim() != 1:
        raise ValueError("z_centers must have shape (Nz,)")
    if anchor_sizes.dim() != 2 or anchor_sizes.shape[1] != 3:
        raise ValueError("anchor_sizes must have shape (M, 3)")

    xx, zz = torch.meshgrid(x_centers, z_centers, indexing="ij")
    centers_xz = torch.stack([xx.reshape(-1), zz.reshape(-1)], dim=1)  # (Nx*Nz, 2)

    num_centers = centers_xz.shape[0]
    num_sizes = anchor_sizes.shape[0]

    centers_xz = centers_xz.repeat_interleave(num_sizes, dim=0)

    centers_y = torch.full(
        (centers_xz.shape[0], 1),
        float(y_center),
        dtype=centers_xz.dtype,
        device=centers_xz.device
    )

    sizes = anchor_sizes.repeat(num_centers, 1)

    # 先用 0 yaw 作为默认朝向，后续可扩展成多朝向 anchor
    yaw = torch.zeros(
        (centers_xz.shape[0], 1),
        dtype=centers_xz.dtype,
        device=centers_xz.device
    )

    anchors_3d = torch.cat(
        [
            centers_xz[:, 0:1],   # x
            centers_y,            # y
            centers_xz[:, 1:2],   # z
            sizes,                # dx, dy, dz
            yaw                   # yaw
        ],
        dim=1
    )

    return anchors_3d


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("generate_3d_anchor_grid", _generate_3d_anchor_grid_impl)


def _generate_3d_anchor_grid_meta(
    x_centers: torch.Tensor,
    z_centers: torch.Tensor,
    y_center: float,
    anchor_sizes: torch.Tensor
) -> torch.Tensor:
    if x_centers.dim() != 1:
        raise ValueError("x_centers must have shape (Nx,)")
    if z_centers.dim() != 1:
        raise ValueError("z_centers must have shape (Nz,)")
    if anchor_sizes.dim() != 2 or anchor_sizes.shape[1] != 3:
        raise ValueError("anchor_sizes must have shape (M, 3)")

    num_anchors = x_centers.shape[0] * z_centers.shape[0] * anchor_sizes.shape[0]
    return torch.empty((num_anchors, 7), device="meta", dtype=x_centers.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("generate_3d_anchor_grid", _generate_3d_anchor_grid_meta)


def generate_3d_anchor_grid(
    x_centers: torch.Tensor,
    z_centers: torch.Tensor,
    y_center: float,
    anchor_sizes: torch.Tensor
) -> torch.Tensor:
    return torch.ops.fusionrepo.generate_3d_anchor_grid(
        x_centers,
        z_centers,
        float(y_center),
        anchor_sizes
    )