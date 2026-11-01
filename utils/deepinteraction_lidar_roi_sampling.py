"""F04：解码并扩大 query box 后采样 LiDAR-BEV ROI。"""

from __future__ import annotations

import torch
from torchvision.ops import roi_align

from utils.deepinteraction_decode_boxes import deepinteraction_decode_boxes as decode_boxes
from utils.deepinteraction_lidar_box_corners import deepinteraction_lidar_box_corners as lidar_box_corners


def _impl(lidar_feature, center, height, dimension, rotation, velocity,
          point_cloud_range, voxel_size, out_size_factor: int,
          output_size: int, sampling_ratio: int):
    """输出 ``[B,K,C,S,S]`` 的 LiDAR ROI。

    原 ``PointRCNNBlock`` 解码框后将 ``dims[3:6]`` 原地扩大两倍，取旋转
    八角点的 BEV 外接矩形，并执行 ROIAlignV2(output=7, scale=1,
    sampling_ratio=2)。本实现保持其数学语义而避免修改输入预测张量。
    ROIAlign 仅对 LiDAR 特征可微；预测字段在 decode 前按原源码 detach，
    box/ROI 坐标不可微；越界采样按 torchvision 零填充。
    原源码：``decoder_utils.py:765-841``，``PointRCNNBlock.forward``。
    """
    boxes = decode_boxes(
        center.detach(), height.detach(), dimension.detach(), rotation.detach(),
        velocity.detach(),
        point_cloud_range, voxel_size, out_size_factor
    )
    corners = lidar_box_corners(boxes, dimension_scale=2.0)[..., :2]
    point_cloud_range = point_cloud_range.to(
        device=corners.device, dtype=corners.dtype
    )
    voxel_size = voxel_size.to(device=corners.device, dtype=corners.dtype)
    feature_coordinates = (
        (corners - point_cloud_range[:2]) / (voxel_size[:2] * out_size_factor)
    )
    outputs = []
    for sample in range(lidar_feature.shape[0]):
        coordinates = feature_coordinates[sample]
        zeros = coordinates.new_zeros(coordinates.shape[0])
        rois = torch.stack(
            (zeros, coordinates[..., 0].amin(-1), coordinates[..., 1].amin(-1),
             coordinates[..., 0].amax(-1), coordinates[..., 1].amax(-1)), -1
        )
        outputs.append(
            roi_align(
                lidar_feature[sample:sample + 1], rois,
                output_size=(output_size, output_size), spatial_scale=1.0,
                sampling_ratio=sampling_ratio, aligned=True
            )
        )
    return torch.stack(outputs)


def _meta(lidar_feature, center, height, dimension, rotation, velocity,
          point_cloud_range, voxel_size, out_size_factor: int,
          output_size: int, sampling_ratio: int):
    return lidar_feature.new_empty(
        (lidar_feature.shape[0], center.shape[-1], lidar_feature.shape[1],
         output_size, output_size)
    )


_IMPLEMENTATION = torch.library.Library(
    "fusionrepo", "IMPL", "CompositeExplicitAutograd"
)
_IMPLEMENTATION.impl("deepinteraction_lidar_roi_sampling", _impl)
_META = torch.library.Library("fusionrepo", "IMPL", "Meta")
_META.impl("deepinteraction_lidar_roi_sampling", _meta)


def deepinteraction_lidar_roi_sampling(*args):
    return torch.ops.fusionrepo.deepinteraction_lidar_roi_sampling(*args)
