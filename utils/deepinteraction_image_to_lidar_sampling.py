"""F01：将 pillar 内点投影到多相机并采样图像特征。"""

from __future__ import annotations

import torch

from utils.project_points_to_normalized_image_grid import (
    project_points_to_normalized_image_grid,
)
from utils.sample_features_by_grid_v2 import sample_features_by_grid_v2


def _impl(
    lidar_feature: torch.Tensor,
    image_feature: torch.Tensor,
    pillars: torch.Tensor,
    pillar_num_points: torch.Tensor,
    pillar_coordinates: torch.Tensor,
    lidar2image: torch.Tensor,
    lidar_augmentation: torch.Tensor,
    image_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """执行 MMRI_I2P 中无参数的投影和采样部分。

    Args:
        lidar_feature: ``[B,C,Hb,Wb]``，增强后 LiDAR-BEV 坐标。
        image_feature: ``[B,V,C,Hi,Wi]``，输入图像 stride=4 特征。
        pillars: ``[M,T,D]``，前三维为增强后米制 xyz，允许 non-contiguous。
        pillar_num_points: ``[M]``，每 pillar 的真实点数。
        pillar_coordinates: ``[M,4]``，``(batch,z,y,x)``。
        lidar2image: ``[B,V,4,4]``，增强前 LiDAR 到图像的投影矩阵。
        lidar_augmentation: ``[B,4,4]``，增强前到增强后的齐次变换。
        image_size: ``[B,2]``，输入图像 ``(H,W)``。
    Returns:
        LiDAR query ``[M,1,C]``、图像 token ``[M,T*V,C]``、有效 mask
        ``[M,T*V]``、原 coordinate。图像采样可微；投影/mask/index 不可微。

    严格保留 ``depth>1e-5``、归一化坐标开区间 ``(-1,1)``、padding 点排除、
    ``grid_sample(align_corners=False)`` 和 scatter 前 token 排列语义。
    原源码：``projects/mmdet3d_plugin/models/utils/encoder_utils.py:237-320``，
    ``MMRI_I2P.forward``。
    """
    batch, channels, _, _ = lidar_feature.shape
    _, views, image_channels, _, _ = image_feature.shape
    pillars = pillars.contiguous()
    m, max_points = pillars.shape[:2]
    query = lidar_feature.new_zeros((m, 1, channels))
    tokens = image_feature.new_zeros((m, max_points * views, image_channels))
    valid = torch.zeros((m, max_points * views), dtype=torch.bool, device=pillars.device)

    for batch_index in range(batch):
        selected = torch.nonzero(
            pillar_coordinates[:, 0] == batch_index, as_tuple=False
        ).flatten()
        if selected.numel() == 0:
            continue
        coordinates = pillar_coordinates[selected]
        query[selected, 0] = lidar_feature[
            batch_index,
            :,
            coordinates[:, 2].long(),
            coordinates[:, 3].long(),
        ].transpose(0, 1)

        identity_image_aug = torch.eye(
            4, device=pillars.device, dtype=pillars.dtype
        ).reshape(1, 4, 4)
        grid, camera_valid = project_points_to_normalized_image_grid(
            pillars[selected, :, :3].detach().unsqueeze(0),
            lidar2image[batch_index:batch_index + 1].to(
                device=pillars.device, dtype=pillars.dtype
            ),
            identity_image_aug,
            lidar_augmentation[batch_index:batch_index + 1].to(
                device=pillars.device, dtype=pillars.dtype
            ),
            image_size[batch_index:batch_index + 1],
            False,
        )
        grid = grid[0]
        camera_valid = camera_valid[0]
        real_point = (
            torch.arange(max_points, device=pillars.device)[None, :]
            < pillar_num_points[selected, None]
        )
        camera_valid &= real_point.unsqueeze(0)
        sampled = sample_features_by_grid_v2(
            image_feature[batch_index:batch_index + 1],
            grid.unsqueeze(0), None, False, "zeros"
        )[0].permute(2, 3, 0, 1)
        tokens[selected] = sampled.reshape(selected.numel(), max_points * views, image_channels)
        valid[selected] = camera_valid.permute(1, 2, 0).reshape(
            selected.numel(), max_points * views
        )
    return query, tokens, valid, pillar_coordinates


def _meta(lidar_feature, image_feature, pillars, pillar_num_points,
          pillar_coordinates, lidar2image, lidar_augmentation, image_size):
    m, points = pillars.shape[:2]
    views = image_feature.shape[1]
    return (
        lidar_feature.new_empty((m, 1, lidar_feature.shape[1])),
        image_feature.new_empty((m, points * views, image_feature.shape[2])),
        torch.empty((m, points * views), device="meta", dtype=torch.bool),
        pillar_coordinates.new_empty(pillar_coordinates.shape),
    )


_IMPLEMENTATION = torch.library.Library(
    "fusionrepo", "IMPL", "CompositeExplicitAutograd"
)
_IMPLEMENTATION.impl("deepinteraction_image_to_lidar_sampling", _impl)
_META = torch.library.Library("fusionrepo", "IMPL", "Meta")
_META.impl("deepinteraction_image_to_lidar_sampling", _meta)


def deepinteraction_image_to_lidar_sampling(*args):
    """Python wrapper；framework 固定通过 torch.ops 调用本算子。"""
    return torch.ops.fusionrepo.deepinteraction_image_to_lidar_sampling(*args)
