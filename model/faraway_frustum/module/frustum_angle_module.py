# -*- coding: utf-8 -*-
"""Faraway-Frustum 视锥 center-view 角度模块。"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrustumAngleModule(nn.Module):
    """根据 2D box 中心与 KITTI P2 标定计算 center-view 旋转角。

    Inputs:
        boxes_2d: ``(K,4)``，图像框 ``(x1,y1,x2,y2)``，像素坐标。
        p2: ``(3,4)``，KITTI image_2 投影矩阵。

    Returns:
        rot_angles: ``(K,)``，绕 camera Y 轴的 center-view 旋转角，单位弧度。
        该角度与 Frustum-ConvNet 的 ``pi/2 + frustum_angle`` 等价；
        使用它旋转相机坐标后，视锥中心方向被规范到正 z 轴附近。
    """

    def __init__(self, reference_depth: float = 20.0):
        super().__init__()
        self.reference_depth = float(reference_depth)

    def forward(self, boxes_2d: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        if boxes_2d.numel() == 0:
            return boxes_2d.new_zeros((0,))
        if boxes_2d.dim() != 2 or boxes_2d.shape[1] != 4:
            raise ValueError("boxes_2d must have shape (K,4)")
        if p2.shape != (3, 4):
            raise ValueError("p2 must have shape (3,4)")
        p2 = p2.to(device=boxes_2d.device, dtype=boxes_2d.dtype)
        u_center = (boxes_2d[:, 0] + boxes_2d[:, 2]) * 0.5
        z_ref = boxes_2d.new_full((boxes_2d.shape[0],), self.reference_depth)
        fx = p2[0, 0].clamp_min(1e-6)
        cx = p2[0, 2]
        tx = p2[0, 3]
        x_rect = (u_center * z_ref - cx * z_ref - tx) / fx
        return torch.atan2(x_rect, z_ref)
