# -*- coding: utf-8 -*-
"""DeepInteraction LiDAR bottom-center box 八角点算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _impl(boxes: torch.Tensor, dimension_scale: float = 1.0) -> torch.Tensor:
    """生成 LiDAR box 八角点。

    Args:
        boxes: ``[...,>=7]``，米制 ``(x,y,bottom_z,dx,dy,dz,yaw,...)``。
        dimension_scale: 仅缩放 ``dx,dy,dz``，bottom center 与 yaw 不变。
    Returns:
        ``[...,8,3]``，dtype/device 与 boxes 一致；角点顺序与
        mmdet3d ``LiDARInstance3DBoxes.corners`` 一致，坐标为右手 LiDAR 系。

    空 box、非 contiguous 输入均按广播规则处理。原语义：mmdet3d
    bottom-center box 约定及 ``decoder_utils.py:765-841``。
    """
    template = boxes.new_tensor([
        [-.5, -.5, 0], [-.5, -.5, 1], [-.5, .5, 1], [-.5, .5, 0],
        [.5, -.5, 0], [.5, -.5, 1], [.5, .5, 1], [.5, .5, 0],
    ])
    local = (template.view(*([1] * (boxes.dim() - 1)), 8, 3)
             * boxes[..., 3:6].unsqueeze(-2) * dimension_scale)
    cosine, sine = torch.cos(boxes[..., 6]), torch.sin(boxes[..., 6])
    x = local[..., 0] * cosine.unsqueeze(-1) - local[..., 1] * sine.unsqueeze(-1)
    y = local[..., 0] * sine.unsqueeze(-1) + local[..., 1] * cosine.unsqueeze(-1)
    return torch.stack((x, y, local[..., 2]), -1) + boxes[..., :3].unsqueeze(-2)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("deepinteraction_lidar_box_corners", _impl)


def _meta(boxes: torch.Tensor, dimension_scale: float = 1.0):
    return boxes.new_empty((*boxes.shape[:-1], 8, 3))


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("deepinteraction_lidar_box_corners", _meta)


def deepinteraction_lidar_box_corners(boxes: torch.Tensor,
                                      dimension_scale: float = 1.0):
    """调用 ``torch.ops.fusionrepo.deepinteraction_lidar_box_corners``。"""
    return torch.ops.fusionrepo.deepinteraction_lidar_box_corners(
        boxes, dimension_scale
    )
