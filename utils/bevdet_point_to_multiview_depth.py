# -*- coding: utf-8 -*-
"""BEVDet 点云投影与多视角最近深度图生成算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _point_to_multiview_depth_impl(
    points: torch.Tensor,
    ego2img: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    image_size: torch.Tensor,
    depth_min: float,
    depth_max: float,
    downsample: int,
) -> torch.Tensor:
    """
    将 ego 点云投影到多个相机平面，为每个像素保留最近深度。

    Args:
        points: ego 点云，shape 为 ``(B, P, C)``。
        ego2img: ego 到图像的投影矩阵，shape 为 ``(B, N, 4, 4)``。
        post_rots: 图像增强旋转，shape 为 ``(B, N, 3, 3)``。
        post_trans: 图像增强平移，shape 为 ``(B, N, 3)``。
        image_size: 增强后图像尺寸 ``(height, width)``。
        depth_min: 有效深度下界。
        depth_max: 有效深度上界。
        downsample: 输出深度图下采样倍数。

    Returns:
        Tensor: 多视角深度图，shape 为 ``(B, N, H/downsample, W/downsample)``。
    """
    bsz, num_points = points.shape[:2]
    num_cams = ego2img.shape[1]
    out_h = int(image_size[0].item()) // int(downsample)
    out_w = int(image_size[1].item()) // int(downsample)

    expanded_points = points[..., :3].unsqueeze(1).expand(bsz, num_cams, num_points, 3)
    flat_points = expanded_points.reshape(bsz * num_cams, num_points, 3)
    flat_matrix = ego2img.reshape(bsz * num_cams, 4, 4).to(points.dtype)
    pixels = torch.ops.fusionrepo.points_to_image(flat_points, flat_matrix[:, :3], 1e-5)
    ones = torch.ones((bsz * num_cams, num_points, 1), device=points.device, dtype=points.dtype)
    homogeneous = torch.cat([flat_points, ones], dim=-1)
    projected = torch.matmul(homogeneous, flat_matrix.transpose(1, 2))
    uv_depth = torch.cat([pixels, projected[..., 2:3]], dim=-1)
    uv_depth = uv_depth.reshape(bsz, num_cams, num_points, 3)
    uv_depth = torch.matmul(post_rots.to(points.dtype)[:, :, None], uv_depth[..., None]).squeeze(-1)
    uv_depth = uv_depth + post_trans.to(points.dtype)[:, :, None]

    pix = torch.round(uv_depth[..., :2] / float(downsample)).long()
    valid = (
        (pix[..., 0] >= 0)
        & (pix[..., 0] < out_w)
        & (pix[..., 1] >= 0)
        & (pix[..., 1] < out_h)
        & (uv_depth[..., 2] >= float(depth_min))
        & (uv_depth[..., 2] < float(depth_max))
    )
    depth_map = points.new_zeros((bsz, num_cams, out_h, out_w))
    for batch_idx in range(bsz):
        for camera_idx in range(num_cams):
            point_ids = torch.nonzero(valid[batch_idx, camera_idx], as_tuple=False).squeeze(1)
            if point_ids.numel() == 0:
                continue
            xy = pix[batch_idx, camera_idx, point_ids]
            depth = uv_depth[batch_idx, camera_idx, point_ids, 2]
            linear = xy[:, 1] * out_w + xy[:, 0]
            nearest = depth.new_full((out_h * out_w,), float("inf"))
            nearest.scatter_reduce_(0, linear, depth, reduce="amin", include_self=True)
            nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
            depth_map[batch_idx, camera_idx] = nearest.view(out_h, out_w)
    return depth_map.contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("point_to_multiview_depth", _point_to_multiview_depth_impl)


def _point_to_multiview_depth_meta(
    points, ego2img, post_rots, post_trans, image_size, depth_min, depth_max, downsample
):
    out_h = int(image_size[0].item()) // int(downsample)
    out_w = int(image_size[1].item()) // int(downsample)
    return torch.empty((points.shape[0], ego2img.shape[1], out_h, out_w), device="meta", dtype=points.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("point_to_multiview_depth", _point_to_multiview_depth_meta)


def point_to_multiview_depth(
    points, ego2img, post_rots, post_trans, image_size, depth_min, depth_max, downsample
):
    """调用 ``fusionrepo::point_to_multiview_depth``。"""
    return torch.ops.fusionrepo.point_to_multiview_depth(
        points, ego2img, post_rots, post_trans, image_size,
        float(depth_min), float(depth_max), int(downsample)
    )
