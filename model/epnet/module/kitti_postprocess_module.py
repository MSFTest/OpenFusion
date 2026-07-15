# -*- coding: utf-8 -*-
"""KITTI label_2 post-processing for EPNet."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class KITTIPostProcessModule(nn.Module):
    """
    将 LiDAR 3D 检测结果转换为 KITTI label_2 文本行。

    Inputs:
        boxes_3d: (M, 7)，LiDAR 坐标 [x, y, z, dx, dy, dz, yaw]。
        scores_3d: (M,)，检测分数，必须来自真实 detector 或其融合修正。
        labels_3d: (M,)，类别索引。
        calib: dict，包含 P2(3,4)、R0_rect(3,3)、Tr_velo2cam(3,4)。

    Returns:
        list[str]，每行格式为:
        type truncated occluded alpha bbox h w l x y z ry score。
    """

    def __init__(self, class_names: Sequence[str], score_thr: float = 0.05):
        super().__init__()
        self.class_names = list(class_names)
        self.score_thr = float(score_thr)

    @staticmethod
    def _lidar_corners(box: torch.Tensor) -> torch.Tensor:
        x, y, z, dx, dy, dz, yaw = [float(v) for v in box.tolist()]
        # MMDetection3D 的 LiDARInstance3DBoxes 在 KITTI 中通常采用 bottom-center
        # origin=(0.5, 0.5, 0)。KITTI label_2 的 location 也是相机坐标底面中心。
        # 因此角点 z 方向应为 [0, dz]，不能按几何中心写成 [-dz/2, dz/2]。
        corners = torch.tensor(
            [
                [dx / 2, dy / 2, dz],
                [dx / 2, -dy / 2, dz],
                [-dx / 2, -dy / 2, dz],
                [-dx / 2, dy / 2, dz],
                [dx / 2, dy / 2, 0.0],
                [dx / 2, -dy / 2, 0.0],
                [-dx / 2, -dy / 2, 0.0],
                [-dx / 2, dy / 2, 0.0],
            ],
            dtype=box.dtype,
            device=box.device,
        )
        c = math.cos(yaw)
        s = math.sin(yaw)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=box.dtype, device=box.device)
        return corners @ rot.T + box.new_tensor([x, y, z])

    @staticmethod
    def _to_4x4(mat: torch.Tensor) -> torch.Tensor:
        out = mat.new_zeros((4, 4))
        out[3, 3] = 1.0
        out[: mat.shape[0], : mat.shape[1]] = mat
        return out

    def _box_to_camera_and_bbox(self, box: torch.Tensor, calib: Dict[str, torch.Tensor]):
        p2 = calib["P2"].to(device=box.device, dtype=box.dtype)
        r0 = self._to_4x4(calib["R0_rect"].to(device=box.device, dtype=box.dtype))
        tr = self._to_4x4(calib["Tr_velo2cam"].to(device=box.device, dtype=box.dtype))
        lidar_to_rect = r0 @ tr

        center = torch.cat([box[:3], box.new_ones(1)])
        center_cam = (lidar_to_rect @ center)[:3]
        corners = self._lidar_corners(box)
        corners_h = torch.cat([corners, torch.ones((8, 1), dtype=box.dtype, device=box.device)], dim=1)
        corners_cam_h = (lidar_to_rect @ corners_h.T).T
        proj = (p2 @ corners_cam_h.T).T
        depth = proj[:, 2].clamp(min=1e-5)
        uv = proj[:, :2] / depth.unsqueeze(1)
        x1, y1 = uv.min(dim=0).values
        x2, y2 = uv.max(dim=0).values

        dx, dy, dz = [float(v) for v in box[3:6].tolist()]
        h, w, l = dz, dy, dx
        yaw = float(box[6].item())
        ry = _normalize_angle(-yaw - math.pi / 2.0)
        alpha = _normalize_angle(ry - math.atan2(float(center_cam[0]), float(center_cam[2])))
        return center_cam, (float(x1), float(y1), float(x2), float(y2)), (h, w, l), ry, alpha

    def forward(self, prediction: Dict[str, torch.Tensor], calib: Dict[str, torch.Tensor]) -> List[str]:
        boxes = prediction["boxes_3d"].detach().cpu()
        scores = prediction["scores_3d"].detach().cpu()
        labels = prediction["labels_3d"].detach().cpu().long()
        lines: List[str] = []
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < self.score_thr:
                continue
            label_idx = int(label.item())
            if label_idx < 0 or label_idx >= len(self.class_names):
                continue
            center_cam, bbox, dims, ry, alpha = self._box_to_camera_and_bbox(box, calib)
            x1, y1, x2, y2 = bbox
            if abs(x2 - x1) < 1e-4 or abs(y2 - y1) < 1e-4:
                continue
            h, w, l = dims
            lines.append(
                f"{self.class_names[label_idx]} 0.00 0 {alpha:.4f} "
                f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} "
                f"{h:.4f} {w:.4f} {l:.4f} "
                f"{float(center_cam[0]):.4f} {float(center_cam[1]):.4f} {float(center_cam[2]):.4f} "
                f"{ry:.4f} {float(score):.6f}"
            )
        return lines
