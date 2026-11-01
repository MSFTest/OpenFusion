# -*- coding: utf-8 -*-
"""Pure NumPy KITTI projection and compact PointPainting score encoding."""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np


# PointPainting channels: background, vehicle, pedestrian, cyclist.
COCO_TO_PAINTING: Mapping[int, int] = {
    0: 2,  # person
    1: 3,  # bicycle
    2: 1,  # car
    3: 3,  # motorcycle
    5: 1,  # bus
    7: 1,  # truck
}


def _as_4x4(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3, :4] = matrix
        return result
    if matrix.shape == (3, 3):
        result = np.eye(4, dtype=np.float32)
        result[:3, :3] = matrix
        return result
    raise ValueError(f"calibration matrix must be 3x3, 3x4, or 4x4, got {matrix.shape}")


def project_lidar_to_image(
    points: np.ndarray,
    tr_velo_to_cam: np.ndarray,
    r0_rect: np.ndarray,
    p2: np.ndarray,
    image_shape: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points while preserving the original point ordering."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape (N, >=3), got {points.shape}")
    xyz1 = np.concatenate(
        [points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
    )
    camera = xyz1 @ (_as_4x4(r0_rect) @ _as_4x4(tr_velo_to_cam)).T
    projected = camera @ _as_4x4(p2).T
    depth = projected[:, 2]
    pixels = np.zeros((points.shape[0], 2), dtype=np.float32)
    positive_depth = depth > 1e-6
    pixels[positive_depth] = projected[positive_depth, :2] / depth[positive_depth, None]
    height, width = int(image_shape[0]), int(image_shape[1])
    valid = (
        positive_depth
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels, valid


def encode_point_semantics(
    points: np.ndarray,
    tr_velo_to_cam: np.ndarray,
    r0_rect: np.ndarray,
    p2: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Encode per-point semantic scores as compact uint8 values."""
    pixels, valid = project_lidar_to_image(
        points, tr_velo_to_cam, r0_rect, p2, image_shape
    )
    masks = np.asarray(masks, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    confidences = np.asarray(confidences, dtype=np.float32)
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (K,H,W), got {masks.shape}")
    if not (masks.shape[0] == labels.shape[0] == confidences.shape[0]):
        raise ValueError("masks, labels, and confidences must have equal lengths")

    semantic = np.zeros((points.shape[0], 4), dtype=np.float32)
    semantic[:, 0] = 1.0
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        xy = np.rint(pixels[valid_indices]).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, image_shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, image_shape[0] - 1)
        for mask, label, confidence in zip(masks, labels, confidences):
            channel = COCO_TO_PAINTING.get(int(label))
            if channel is None:
                continue
            hit = mask[xy[:, 1], xy[:, 0]]
            hit_indices = valid_indices[hit]
            semantic[hit_indices, channel] = np.maximum(
                semantic[hit_indices, channel], np.clip(float(confidence), 0.0, 1.0)
            )
        semantic[valid_indices, 0] = 1.0 - semantic[valid_indices, 1:].max(axis=1)
    return np.rint(np.clip(semantic, 0.0, 1.0) * 255.0).astype(np.uint8)
