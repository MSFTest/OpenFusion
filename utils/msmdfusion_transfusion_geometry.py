"""TransFusion 私有几何依赖。来源：mmdet3d/core/bbox/structures/
base_box3d.py BaseInstance3DBoxes.height_overlaps 352、overlaps 384；
mmdet3d/core/bbox/iou_calculators/iou3d_calculator.py bbox_overlaps_3d 141；
mmdet3d/core/utils/gaussian.py gaussian_2d 5、draw_heatmap_gaussian 24、
gaussian_radius 56；原 CUDA boxes_overlap_bev_gpu 的旋转矩形交面积公式。
"""

from __future__ import annotations

import math
import numpy as np
import torch


def _corners(box: torch.Tensor) -> list[tuple[float, float]]:
    """LiDAR 底面 (x,y,dx,dy,yaw) 的逆时针四角。"""
    x, y, _, dx, dy, _, yaw = [float(v) for v in box[:7]]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + u * c - v * s, y + u * s + v * c)
            for u, v in ((-dx/2, -dy/2), (dx/2, -dy/2),
                         (dx/2, dy/2), (-dx/2, dy/2))]


def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _intersection(a, b, p, q):
    cp, cq = _cross(a, b, p), _cross(a, b, q)
    factor = cp / (cp - cq)
    return (p[0] + factor * (q[0] - p[0]), p[1] + factor * (q[1] - p[1]))


def _rectangle_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """凸多边形裁剪求旋转 BEV 交面积；原 CUDA kernel 取同一几何交集。"""
    polygon = _corners(a)
    clip = _corners(b)
    for edge in range(4):
        first, second = clip[edge], clip[(edge + 1) % 4]
        if not polygon:
            return 0.0
        before, polygon_new = polygon[-1], []
        for current in polygon:
            inside_before = _cross(first, second, before) >= 0
            inside_current = _cross(first, second, current) >= 0
            if inside_current != inside_before:
                polygon_new.append(_intersection(first, second, before, current))
            if inside_current:
                polygon_new.append(current)
            before = current
        polygon = polygon_new
    if len(polygon) < 3:
        return 0.0
    twice_area = sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
                     - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
                     for i in range(len(polygon)))
    return abs(twice_area) * 0.5


def box3d_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """原 BboxOverlaps3D：BEV 交面积 × z 交高 / 并体积。

    原 Hungarian 路径的预测框已经 detach，IoU 只进入离散匹配；
    因此参考计算不提供梯度。保留空矩阵及分母下界 1e-8。
    """
    result = boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    for i, a in enumerate(boxes1):
        for j, b in enumerate(boxes2):
            overlap_xy = _rectangle_overlap(a, b)
            overlap_z = max(0.0, min(float(a[2] + a[5]), float(b[2] + b[5]))
                            - max(float(a[2]), float(b[2])))
            intersection = overlap_xy * overlap_z
            volume_a = float(a[3] * a[4] * a[5])
            volume_b = float(b[3] * b[4] * b[5])
            result[i, j] = intersection / max(volume_a + volume_b - intersection, 1e-8)
    return result


def gaussian_radius(det_size, min_overlap=0.5):
    """原 mmdet3d/core/utils/gaussian.py 56–85 行。"""
    height, width = det_size
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + torch.sqrt(b1**2 - 4 * c1)) / 2
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + torch.sqrt(b2**2 - 16 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + torch.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """原 mmdet3d/core/utils/gaussian.py 5–53 行，截断后逐像素取大值。"""
    diameter = 2 * radius + 1
    middle = (diameter - 1) / 2
    y, x = np.ogrid[-middle:middle + 1, -middle:middle + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * (diameter / 6) ** 2))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    cx, cy = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    local = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    sampled = torch.from_numpy(gaussian[radius - top:radius + bottom,
                                         radius - left:radius + right]).to(heatmap.device, torch.float32)
    if min(sampled.shape) > 0 and min(local.shape) > 0:
        torch.max(local, sampled * k, out=local)
    return heatmap
