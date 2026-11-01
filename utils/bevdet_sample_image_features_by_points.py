# -*- coding: utf-8 -*-
"""BEVDet 3D 参考点多视角投影与图像特征采样算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _sample_image_features_by_points_impl(
    ref_points: torch.Tensor,
    image_feats: torch.Tensor,
    sensor2ego: torch.Tensor,
    ego2global: torch.Tensor,
    intrinsics: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
    bda: torch.Tensor,
    img_feat_downsample: int,
) -> torch.Tensor:
    """
    将增强 ego 参考点投影至多相机特征图并选择首个可见相机特征。

    Args:
        ref_points: 增强 ego 坐标点，shape 为 ``(B,K,3)``。
        image_feats: 多相机特征，shape 为 ``(B,N,C,Hf,Wf)``。
        sensor2ego: 相机到 ego 矩阵，shape 为 ``(B,N,4,4)``。
        ego2global: ego 到 global 矩阵，shape 为 ``(B,N,4,4)``，为接口兼容保留。
        intrinsics: 相机内参，shape 为 ``(B,N,3,3)``。
        post_rots: 图像增强旋转，shape 为 ``(B,N,3,3)``。
        post_trans: 图像增强平移，shape 为 ``(B,N,3)``。
        bda: BEV 增强矩阵，shape 为 ``(B,4,4)``。
        img_feat_downsample: 图像特征相对增强图像的下采样倍数。

    Returns:
        Tensor: 首个可见相机采样特征，shape 为 ``(B,C,K)``。
    """
    del ego2global
    bsz, num_points, _ = ref_points.shape
    num_cams, channels = image_feats.shape[1:3]
    feat_h, feat_w = image_feats.shape[-2:]

    ego2cam = torch.linalg.inv(sensor2ego.to(ref_points.dtype))
    identity_aug = torch.eye(4, device=ref_points.device, dtype=ref_points.dtype)
    identity_aug = identity_aug.view(1, 1, 4, 4).expand(bsz, num_cams, 4, 4)
    unit_size = ref_points.new_tensor([1.0, 1.0])
    base_grid, _, depth = torch.ops.fusionrepo.project_points_to_normalized_image_grid(
        ref_points, ego2cam, identity_aug, bda, unit_size, True
    )
    camera_uv = (base_grid + 1.0) * 0.5
    camera_points = torch.cat([camera_uv * depth.unsqueeze(-1), depth.unsqueeze(-1)], dim=-1)
    projected = torch.matmul(
        intrinsics.to(ref_points.dtype)[:, :, None], camera_points[..., None]
    ).squeeze(-1)
    uv = projected[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1)
    uv_depth = torch.cat([uv, depth.unsqueeze(-1)], dim=-1)
    uv_depth = torch.matmul(
        post_rots.to(ref_points.dtype)[:, :, None], uv_depth[..., None]
    ).squeeze(-1) + post_trans.to(ref_points.dtype)[:, :, None]
    uv_feat = uv_depth[..., :2] / float(img_feat_downsample)
    norm_x = uv_feat[..., 0] / max(feat_w - 1, 1) * 2.0 - 1.0
    norm_y = uv_feat[..., 1] / max(feat_h - 1, 1) * 2.0 - 1.0
    grids = torch.stack([norm_x, norm_y], dim=-1)
    valid = (
        (uv_depth[..., 2] > 0.5)
        & (norm_x >= -1.0) & (norm_x <= 1.0)
        & (norm_y >= -1.0) & (norm_y <= 1.0)
    )
    sampled_all = torch.ops.fusionrepo.sample_features_by_grid_v2(
        image_feats, grids, valid, True, "zeros"
    )
    # 原 DAL 用投影横坐标到各相机主点的距离选择视角，而不是简单取首个可见相机。
    depth_valid = uv_depth[..., 2] > 0.5
    projected_x = uv[..., 0] * depth_valid.to(uv.dtype)
    principal_x = intrinsics.to(ref_points.dtype)[:, :, 0, 2].unsqueeze(-1)
    camera_ids = torch.argmin((projected_x - principal_x).abs(), dim=1)
    gather_index = camera_ids[:, None, None, :].expand(bsz, 1, channels, num_points)
    sampled = sampled_all.gather(dim=1, index=gather_index).squeeze(1)
    point_valid = valid.gather(1, camera_ids[:, None, :]).squeeze(1).to(sampled.dtype)
    return (sampled * point_valid[:, None, :]).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("sample_image_features_by_points", _sample_image_features_by_points_impl)


def _sample_image_features_by_points_meta(
    ref_points, image_feats, sensor2ego, ego2global, intrinsics,
    post_rots, post_trans, bda, img_feat_downsample
):
    return torch.empty(
        (ref_points.shape[0], image_feats.shape[2], ref_points.shape[1]),
        device="meta",
        dtype=image_feats.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("sample_image_features_by_points", _sample_image_features_by_points_meta)


def sample_image_features_by_points(
    ref_points, image_feats, sensor2ego, ego2global, intrinsics,
    post_rots, post_trans, bda, img_feat_downsample
):
    """调用 ``fusionrepo::sample_image_features_by_points``。"""
    return torch.ops.fusionrepo.sample_image_features_by_points(
        ref_points, image_feats, sensor2ego, ego2global, intrinsics,
        post_rots, post_trans, bda, int(img_feat_downsample)
    )
