# -*- coding: utf-8 -*-
"""LSS 视锥到增强 ego 坐标系的投影算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _project_frustum_to_ego_impl(
    frustum: torch.Tensor,
    sensor2ego: torch.Tensor,
    intrinsics: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    bda: torch.Tensor,
) -> torch.Tensor:
    """
    将图像视锥经逆图像增强、相机内外参和 BEV 增强投影到 ego。

    Args:
        frustum: ``(u,v,depth)`` 视锥，shape 为 ``(D,H,W,3)``。
        sensor2ego: 相机到 ego 矩阵，shape 为 ``(B,N,4,4)``。
        intrinsics: 相机内参，shape 为 ``(B,N,3,3)``。
        post_rots: 图像增强旋转，shape 为 ``(B,N,3,3)``。
        post_trans: 图像增强平移，shape 为 ``(B,N,3)``。
        bda: BEV 增强矩阵，shape 为 ``(B,3,3)`` 或 ``(B,4,4)``。

    Returns:
        Tensor: 增强 ego 坐标，shape 为 ``(B,N,D,H,W,3)``。
    """
    if frustum.dim() != 4 or frustum.shape[-1] != 3:
        raise ValueError("frustum must have shape (D,H,W,3)")
    bsz, num_cams = sensor2ego.shape[:2]
    points = frustum.to(sensor2ego.dtype).view(1, 1, *frustum.shape)
    points = points.expand(bsz, num_cams, *frustum.shape)
    points = points - post_trans.view(bsz, num_cams, 1, 1, 1, 3)
    points = torch.matmul(
        torch.linalg.inv(post_rots.to(points.dtype)).view(bsz, num_cams, 1, 1, 1, 3, 3),
        points.unsqueeze(-1),
    )
    points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), dim=5)
    cam_to_ego = sensor2ego[:, :, :3, :3].to(points.dtype).matmul(
        torch.linalg.inv(intrinsics.to(points.dtype))
    )
    points = torch.matmul(
        cam_to_ego.view(bsz, num_cams, 1, 1, 1, 3, 3), points
    ).squeeze(-1)
    points = points + sensor2ego[:, :, :3, 3].to(points.dtype).view(bsz, num_cams, 1, 1, 1, 3)
    bda_rot = bda[:, :3, :3].to(points.dtype)
    if bda.shape[-1] == 4:
        bda_trans = bda[:, :3, 3].to(points.dtype)
    else:
        bda_trans = torch.zeros((bsz, 3), device=points.device, dtype=points.dtype)
    points = torch.matmul(
        bda_rot.view(bsz, 1, 1, 1, 1, 3, 3), points.unsqueeze(-1)
    ).squeeze(-1)
    return (points + bda_trans.view(bsz, 1, 1, 1, 1, 3)).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("project_frustum_to_ego", _project_frustum_to_ego_impl)


def _project_frustum_to_ego_meta(frustum, sensor2ego, intrinsics, post_rots, post_trans, bda):
    return torch.empty(
        (sensor2ego.shape[0], sensor2ego.shape[1], *frustum.shape[:-1], 3),
        device="meta",
        dtype=sensor2ego.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("project_frustum_to_ego", _project_frustum_to_ego_meta)


def project_frustum_to_ego(frustum, sensor2ego, intrinsics, post_rots, post_trans, bda):
    """调用 ``fusionrepo::project_frustum_to_ego``。"""
    return torch.ops.fusionrepo.project_frustum_to_ego(
        frustum, sensor2ego, intrinsics, post_rots, post_trans, bda
    )
