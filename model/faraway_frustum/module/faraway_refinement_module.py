# -*- coding: utf-8 -*-
"""Faraway 目标 BEV refinement 模块。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

import utils  # noqa: F401
import common  # noqa: F401
import fusion_unit  # noqa: F401
from common.faraway_frustum_box_regression import FarawayFrustumBoxRegression


def _rotate_camera_y(points: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """按 Frustum-ConvNet center-view 约定旋转 camera x/z 坐标。"""
    cosine = torch.cos(angles).to(dtype=points.dtype)
    sine = torch.sin(angles).to(dtype=points.dtype)
    x = cosine * points[:, 0] + sine * points[:, 2]
    z = -sine * points[:, 0] + cosine * points[:, 2]
    return torch.stack([x, points[:, 1], z], dim=1)


def _inverse_rotate_camera_y(points: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angles).to(dtype=points.dtype)
    sine = torch.sin(angles).to(dtype=points.dtype)
    x = cosine * points[:, 0] - sine * points[:, 2]
    z = sine * points[:, 0] + cosine * points[:, 2]
    return torch.stack([x, points[:, 1], z], dim=1)


def _normalise_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class FarawayRefinementModule(nn.Module):
    """对远距离 frustum 点簇执行局部 BEV 回归。

    坐标系说明：
        输入 ``clusters`` 与 ``centroids`` 均为 KITTI camera rect 坐标。
        模块先按 frustum 角旋转到 center-view 坐标，再平移到旋转后的
        centroid 局部坐标；网络回归 center-view 中的局部框参数，最后反旋转
        回 camera rect 坐标表达的 ``(x,y,z,w,l,h,ry)``。

    Inputs:
        clusters: ``(F, P, 3)``，远距离实例点簇。
        counts: ``(F,)``，每个实例有效点数。
        centroids: ``(F, 3)``，相机坐标 centroid。
        frustum_angles: ``(F,)``，每个 frustum 绕竖直轴的角度。

    Returns:
        dict，包含 ``boxes_3d_camera (F,7)``、``regression_outputs (F,7)``、
        ``bev_maps (F,1,H,W)``。``regression_outputs`` 使用原项目标签格式：
        ``(x_shifted, y, z_shifted, h, w, l, ry)``。
    """

    def __init__(
        self,
        bev_height: int = 300,
        bev_width: int = 200,
        x_range: float = 10.0,
        z_range: float = 15.0,
        bb_shift: float = 75.0,
    ):
        super().__init__()
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)
        self.x_range = float(x_range)
        self.z_range = float(z_range)
        self.bb_shift = float(bb_shift)
        self.ff_net = FarawayFrustumBoxRegression()
        self.score_head = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        clusters: torch.Tensor,
        counts: torch.Tensor,
        centroids: torch.Tensor,
        frustum_angles: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if clusters.shape[0] == 0:
            empty_boxes = clusters.new_zeros((0, 7))
            empty_reg = clusters.new_zeros((0, 7))
            empty_bev = clusters.new_zeros((0, 1, self.bev_height, self.bev_width))
            return {
                "boxes_3d_camera": empty_boxes,
                "regression_outputs": empty_reg,
                "centroids": centroids.new_zeros((0, 3)),
                "score_logits": clusters.new_zeros((0,)),
                "bev_maps": empty_bev,
            }

        aligned = torch.ops.fusionrepo.faraway_frustum_axis_alignment(
            clusters,
            counts,
            -frustum_angles,
        )
        aligned_centroids = _rotate_camera_y(centroids, -frustum_angles)
        local = torch.ops.fusionrepo.faraway_centroid_coordinate_transform(
            aligned,
            counts,
            aligned_centroids,
        )
        bev_maps = torch.ops.fusionrepo.faraway_frustum_bev_projection(
            local,
            counts,
            self.bev_height,
            self.bev_width,
            self.x_range,
            self.z_range,
        )
        raw_outputs = self.ff_net(bev_maps)
        dims = torch.nn.functional.softplus(raw_outputs[:, 3:6]) + 1e-3
        yaw_view = 0.5 * torch.atan2(raw_outputs[:, 6:7], raw_outputs[:, 7:8])
        regression_outputs = torch.cat(
            [
                torch.nn.functional.softplus(raw_outputs[:, 0:1]),
                raw_outputs[:, 1:2],
                torch.nn.functional.softplus(raw_outputs[:, 2:3]),
                dims,
                yaw_view,
            ],
            dim=1,
        )
        # 2D 框中心射线比稀疏点簇的横向直方图质心稳定；x 直接表示
        # center-view 射线残差，z 仍以点簇质心为锚点。
        x_view = regression_outputs[:, 0:1] - self.bb_shift
        y = regression_outputs[:, 1:2]
        z_view = regression_outputs[:, 2:3] - self.bb_shift + aligned_centroids[:, 2:3]
        h = regression_outputs[:, 3:4]
        w = regression_outputs[:, 4:5]
        l = regression_outputs[:, 5:6]
        center_view = torch.cat([x_view, y, z_view], dim=1)
        center_camera = _inverse_rotate_camera_y(center_view, -frustum_angles)
        ry = _normalise_angle(regression_outputs[:, 6:7] + frustum_angles[:, None])
        boxes_3d_camera = torch.cat([center_camera[:, 0:1], center_camera[:, 1:2], center_camera[:, 2:3], w, l, h, ry], dim=1)
        score_features = torch.cat(
            [
                regression_outputs,
                torch.log1p(counts.to(dtype=raw_outputs.dtype)).unsqueeze(1),
                centroids[:, 2:3] / 100.0,
            ],
            dim=1,
        )
        score_logits = self.score_head(score_features).squeeze(1)
        return {
            "boxes_3d_camera": boxes_3d_camera,
            "regression_outputs": regression_outputs,
            "centroids": centroids,
            "aligned_centroids": aligned_centroids,
            "frustum_angles": frustum_angles,
            "score_logits": score_logits,
            "bev_maps": bev_maps,
        }
