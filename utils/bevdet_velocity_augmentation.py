# -*- coding: utf-8 -*-
"""DAL 多 sweep 静止目标速度增强。"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch


class BEVDetVelocityAugmentation:
    """按原 DAL 轨迹插值方式增强静止框及其多 sweep 内部点。

    Args:
        rate: 各候选静止实例执行增强的概率。
        rate_vy: 采样横向速度的概率。
        speed_range: 类别到 ``(vx_min,vx_max,vy_max)`` 的映射。

    Inputs:
        points: 关键 ego 坐标点云，shape ``(P,5)``，第 5 维是时间差秒。
        boxes: 底面中心框，shape ``(M,9)``。
        labels: nuScenes 类别索引，shape ``(M,)``。

    Returns:
        Tuple[Tensor, Tensor]: 同 shape 的增强点云与框。
    """

    CLASSES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
               "motorcycle", "bicycle", "pedestrian", "traffic_cone")
    DEFAULT_SPEED_RANGE = {
        "car": (-10, 30, 6), "truck": (-10, 30, 6),
        "construction_vehicle": (-10, 30, 3), "bus": (-10, 30, 3),
        "trailer": (-10, 30, 3), "barrier": (-5, 5, 3),
        "motorcycle": (-2, 25, 3), "bicycle": (-2, 15, 2),
        "pedestrian": (-1, 10, 2),
    }

    def __init__(self, rate: float = 0.5, rate_vy: float = 0.2,
                 speed_range: Mapping[str, Sequence[float]] = None,
                 threshold_vy_by_vx: float = 1.0, rate_rotation: float = -1.0):
        self.rate = float(rate)
        self.rate_vy = float(rate_vy)
        self.speed_range = dict(self.DEFAULT_SPEED_RANGE if speed_range is None else speed_range)
        self.threshold_vy_by_vx = float(threshold_vy_by_vx)
        self.rate_rotation = float(rate_rotation)

    @staticmethod
    def _interpolate(vx: float, vy: float, delta_t: np.ndarray,
                     box: np.ndarray, rotation_delta: float) -> np.ndarray:
        maximum_time = np.max(delta_t)
        if vy == 0 or vx == 0:
            delta_x, delta_y = delta_t * vx, np.zeros_like(delta_t)
            interpolated_rotation = np.zeros_like(delta_t)
        else:
            theta = np.arctan2(abs(vy), abs(vx))
            rotation = 2 * theta
            radius = 0.5 * maximum_time * np.sqrt(vx ** 2 + vy ** 2) / np.sin(theta)
            interpolated_rotation = delta_t / maximum_time * rotation
            delta_y = radius - radius * np.cos(interpolated_rotation)
            delta_x = radius * np.sin(interpolated_rotation)
            if vy < 0:
                delta_y = -delta_y
            if vx < 0:
                delta_x = -delta_x
            if np.logical_xor(vx > 0, vy > 0):
                interpolated_rotation = -interpolated_rotation
        augmentation = np.zeros((delta_t.shape[0], 3, 3), dtype=np.float32)
        augmentation[:, 2, 2] = 1.0
        sine, cosine = np.sin(-interpolated_rotation), np.cos(-interpolated_rotation)
        augmentation[:, :2, :2] = np.stack([cosine, sine, -sine, cosine], axis=-1).reshape(-1, 2, 2)
        augmentation[:, :2, 2] = np.stack([delta_x, delta_y], axis=-1)
        corner_to_center = np.eye(3)
        corner_to_center[0, 2] = -0.5 * box[3]
        instance_to_ego = np.eye(3)
        sine, cosine = np.sin(-box[6]), np.cos(-box[6])
        instance_to_ego[:2, :2] = np.asarray([cosine, sine, -sine, cosine]).reshape(2, 2)
        instance_to_ego[:2, 2] = box[:2]
        corner_to_ego = (instance_to_ego @ corner_to_center)[None]
        if rotation_delta != 0:
            local_rotation = np.eye(3)
            sine, cosine = np.sin(-rotation_delta), np.cos(-rotation_delta)
            local_rotation[:2, :2] = np.asarray([cosine, sine, -sine, cosine]).reshape(2, 2)
            rotated_instance = instance_to_ego.copy()
            sine, cosine = np.sin(-box[6] - rotation_delta), np.cos(-box[6] - rotation_delta)
            rotated_instance[:2, :2] = np.asarray([cosine, sine, -sine, cosine]).reshape(2, 2)
            rotated_corner = (rotated_instance @ corner_to_center)[None]
            local_rotation = instance_to_ego @ local_rotation @ np.linalg.inv(instance_to_ego)
            return rotated_corner @ augmentation @ np.linalg.inv(rotated_corner) @ local_rotation[None]
        return corner_to_ego @ augmentation @ np.linalg.inv(corner_to_ego)

    def __call__(self, points: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor):
        if boxes.numel() == 0 or points.numel() == 0:
            return points, boxes
        from mmdet3d.structures.ops import box_np_ops
        points_np = points.detach().cpu().numpy().copy()
        boxes_np = boxes.detach().cpu().numpy().copy()
        memberships = box_np_ops.points_in_rbbox(points_np, boxes_np)
        velocity_norm = np.square(boxes_np[:, 7:9]).sum(axis=1)
        for box_index in range(boxes_np.shape[0]):
            name = self.CLASSES[int(labels[box_index])]
            object_points = points_np[memberships[:, box_index]]
            timestamps = np.unique(object_points[:, 4]) if object_points.size else np.asarray([])
            if (object_points.shape[0] == 0 or timestamps.shape[0] < 3 or velocity_norm[box_index] > 0.01
                    or name not in self.speed_range or np.random.rand() > self.rate):
                continue
            bounds = self.speed_range[name]
            vx = -(np.random.rand() * (bounds[1] - bounds[0]) + bounds[0])
            if np.random.rand() < self.rate_vy:
                max_vy = min(bounds[2] * 2, abs(vx) * self.threshold_vy_by_vx)
                vy = (np.random.rand() - 0.5) * max_vy
            else:
                vy = 0.0
            rotation_delta = ((np.random.rand() - 0.5) * 1.57
                              if np.random.rand() < self.rate_rotation else 0.0)
            transforms = self._interpolate(vx, vy, timestamps, boxes_np[box_index], rotation_delta)
            boxes_np[box_index, 6] += rotation_delta
            center = boxes_np[box_index:box_index + 1, :2]
            last = int(np.argmax(timestamps))
            augmented_center = center @ transforms[last, :2, :2].T + transforms[last, :2, 2]
            boxes_np[box_index, 7:9] = (center - augmented_center) / np.max(timestamps)
            for frame_index, timestamp in enumerate(timestamps):
                selected = object_points[:, 4] == timestamp
                object_points[selected, :2] = (object_points[selected, :2] @ transforms[frame_index, :2, :2].T
                                                + transforms[frame_index, :2, 2])
            points_np[memberships[:, box_index]] = object_points
        return (torch.from_numpy(points_np).to(points), torch.from_numpy(boxes_np).to(boxes))
