"""F02：IP-Basic 深度补全、反投影和 LiDAR-BEV 采样。"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from utils.project_points_to_normalized_image_grid import (
    project_points_to_normalized_image_grid,
)
from utils.sample_features_by_grid_v2 import sample_features_by_grid_v2


_FULL_5 = np.ones((5, 5), np.uint8)
_FULL_9 = np.ones((9, 9), np.uint8)
_CROSS_3 = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
_CROSS_5 = np.asarray(
    [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1],
     [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], np.uint8
)
_CROSS_7 = np.asarray(
    [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0]], np.uint8
)


def _fill_in_multiscale(depth_map: np.ndarray) -> np.ndarray:
    """原 ``ip_basic/depth_map_utils.py:134-287`` 的 multiscale 路径。"""
    depths = np.float32(depth_map)
    near = (depths > .1) & (depths <= 15.0)
    medium = (depths > 15.0) & (depths <= 30.0)
    far = depths > 30.0
    inverted = depths.copy()
    present = inverted > .1
    inverted[present] = 100.0 - inverted[present]
    dilated_far = cv2.dilate(inverted * far, _CROSS_3)
    dilated_medium = cv2.dilate(inverted * medium, _CROSS_5)
    dilated_near = cv2.dilate(inverted * near, _CROSS_7)
    merged = inverted.copy()
    for candidate in (dilated_far, dilated_medium, dilated_near):
        mask = candidate > .1
        merged[mask] = candidate[mask]
    closed = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, _FULL_5)
    median = cv2.medianBlur(closed, 5)
    blurred = closed.copy()
    present = closed > .1
    blurred[present] = median[present]
    top_mask = np.ones(depths.shape, dtype=bool)
    for column in range(blurred.shape[1]):
        first = np.argmax(blurred[:, column] > .1)
        top_mask[:first, column] = False
    empty = (blurred <= .1) & top_mask
    dilated = cv2.dilate(blurred, _FULL_9)
    filled = blurred.copy()
    filled[empty] = dilated[empty]
    # 原 IP-Basic 在 9x9 hole fill 后以 s5 结果重新计算 top mask。past 版本
    # 曾错误复用 s4 mask，会把本应位于新 top row 之上的区域继续扩张填充。
    top_mask = np.ones(filled.shape, dtype=bool)
    top_rows = np.argmax(filled > .1, axis=0)
    for column, first in enumerate(top_rows):
        top_mask[:first, column] = False
    for _ in range(6):
        empty = (filled < .1) & top_mask
        dilated = cv2.dilate(filled, _FULL_5)
        filled[empty] = dilated[empty]
    median = cv2.medianBlur(filled, 5)
    present = (filled > .1) & top_mask
    filled[present] = median[present]
    bilateral = cv2.bilateralFilter(filled, 5, .5, 2.0)
    filled[present] = bilateral[present]
    present = filled > .1
    filled[present] = 100.0 - filled[present]
    return filled


def _impl(
    lidar_feature: torch.Tensor,
    image_feature_size: torch.Tensor,
    points: torch.Tensor,
    points_num: torch.Tensor,
    lidar2image: torch.Tensor,
    lidar_augmentation: torch.Tensor,
    image_size: torch.Tensor,
    point_cloud_range: torch.Tensor,
) -> torch.Tensor:
    """生成图像像素对应的 LiDAR-BEV 特征 ``[B,V,C,Hi,Wi]``。

    Args:
        lidar_feature: ``[B,C,Hb,Wb]``。
        image_feature_size: 含数据值 ``[Hi,Wi]`` 的 int tensor。
        points: ``[B,P,D]``，增强后 LiDAR 点；points_num 排除 padding。
        lidar2image/lidar_augmentation/image_size: 与 F01 含义一致。
        point_cloud_range: ``[6]``，米制增强后 LiDAR 范围。
    Returns:
        反投影并采样的 ``[B,V,C,Hi,Wi]``。仅对 lidar_feature 可微；原实现
        在深度补全前显式 detach/CPU/NumPy，故深度与几何路径不可微。

    同像素点采用原高级索引赋值语义；范围判断均为开区间；最终使用
    ``grid_sample(align_corners=False)`` 并将范围外像素清零。
    原源码：``encoder_utils.py:137-199`` ``BEVWarp.forward``；深度补全：
    ``ip_basic/depth_map_utils.py:134-287``。
    """
    if lidar_feature.device.type == "meta" or image_feature_size.device.type == "meta":
        raise RuntimeError(
            "deepinteraction_lidar_to_image_sampling has data-dependent output "
            "shape and intentionally has no Meta kernel"
        )
    feature_h, feature_w = (int(x) for x in image_feature_size.tolist())
    batch, channels = lidar_feature.shape[:2]
    views = lidar2image.shape[1]
    outputs = []
    for batch_index in range(batch):
        raw_h, raw_w = image_size[batch_index].tolist()
        xyz = points[batch_index, : int(points_num[batch_index]), :3].detach()
        identity_image_aug = torch.eye(
            4, device=xyz.device, dtype=xyz.dtype
        ).reshape(1, 4, 4)
        normalized, visible, depth = project_points_to_normalized_image_grid(
            xyz.unsqueeze(0), lidar2image[batch_index:batch_index + 1].to(
                device=xyz.device, dtype=xyz.dtype
            ),
            identity_image_aug,
            lidar_augmentation[batch_index:batch_index + 1].to(
                device=xyz.device, dtype=xyz.dtype
            ),
            image_size[batch_index:batch_index + 1], True,
        )
        normalized, visible, depth = normalized[0], visible[0], depth[0]
        xy = torch.stack(
            ((normalized[..., 0] + 1) * raw_w * .5,
             (normalized[..., 1] + 1) * raw_h * .5), dim=-1
        )
        augmentation = lidar_augmentation[batch_index].to(
            device=xyz.device, dtype=xyz.dtype
        )
        projection = lidar2image[batch_index].to(
            device=xyz.device, dtype=xyz.dtype
        )
        depth_map = lidar_feature.new_zeros((views, feature_h, feature_w))
        for view in range(views):
            chosen = torch.nonzero(visible[view], as_tuple=False).flatten()
            if chosen.numel():
                px = (xy[view, chosen, 0] / raw_w * feature_w).long()
                py = (xy[view, chosen, 1] / raw_h * feature_h).long()
                depth_map[view, py, px] = depth[view, chosen]
            completed = _fill_in_multiscale(depth_map[view].detach().cpu().numpy())
            depth_map[view] = depth_map.new_tensor(completed)
        xs = torch.linspace(0, raw_w - 1, feature_w, device=lidar_feature.device)
        ys = torch.linspace(0, raw_h - 1, feature_h, device=lidar_feature.device)
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        xs = xs.unsqueeze(0).expand(views, -1, -1)
        ys = ys.unsqueeze(0).expand(views, -1, -1)
        xyd = torch.stack(
            (xs * depth_map, ys * depth_map, depth_map, torch.ones_like(depth_map)), -1
        )
        raw_xyz = torch.einsum(
            "vij,vhwj->vhwi", torch.linalg.inv(projection), xyd
        )[..., :3]
        raw_xyz_h = torch.cat((raw_xyz, torch.ones_like(raw_xyz[..., :1])), -1)
        augmented_xyz = torch.einsum(
            "ij,vhwj->vhwi", augmentation, raw_xyz_h
        )[..., :3].detach()
        current_range = point_cloud_range.to(
            device=augmented_xyz.device, dtype=augmented_xyz.dtype
        )
        inside = (
            (augmented_xyz[..., 0] > current_range[0])
            & (augmented_xyz[..., 0] < current_range[3])
            & (augmented_xyz[..., 1] > current_range[1])
            & (augmented_xyz[..., 1] < current_range[4])
            & (augmented_xyz[..., 2] > current_range[2])
            & (augmented_xyz[..., 2] < current_range[5])
        )
        grid = (
            (augmented_xyz[..., :2] - current_range[:2])
            / (current_range[3:5] - current_range[:2]) * 2 - 1
        )
        source = lidar_feature[batch_index:batch_index + 1].unsqueeze(1).expand(
            -1, views, -1, -1, -1
        )
        sampled = sample_features_by_grid_v2(
            source, grid.unsqueeze(0), None, False, "zeros"
        )[0]
        outputs.append(sampled.masked_fill(~inside.unsqueeze(1), 0))
    return torch.stack(outputs)


_IMPLEMENTATION = torch.library.Library(
    "fusionrepo", "IMPL", "CompositeExplicitAutograd"
)
_IMPLEMENTATION.impl("deepinteraction_lidar_to_image_sampling", _impl)
# 输出 H/W 依赖 image_feature_size 的数据值，当前 PyTorch Meta/FakeTensor 无法
# 准确符号化；按约束不注册伪造 shape 的 Meta kernel。


def deepinteraction_lidar_to_image_sampling(*args):
    """Python wrapper。"""
    return torch.ops.fusionrepo.deepinteraction_lidar_to_image_sampling(*args)
