# -*- coding: utf-8 -*-
"""
KITTIResultModule

将检测结果整理为 KITTI label_2 文本格式。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class KITTIResultModule(nn.Module):
    """
    KITTI 结果格式化模块。

    输入:
        detector_output: 点云接口返回 dict。

    输出:
        list[str]，每一项为 KITTI label_2 格式一行。
    """

    def __init__(self, class_names: tuple[str, ...] = ("Pedestrian", "Cyclist", "Car")):
        super().__init__()
        self.class_names = tuple(class_names)

    def _box_lidar_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        x, y, z, dx, dy, dz, yaw = boxes.unbind(dim=1)
        template = boxes.new_tensor(
            [
                [0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ]
        )
        dims = torch.stack([dx, dy, dz], dim=1)
        corners = template.unsqueeze(0) * dims.unsqueeze(1)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot = boxes.new_zeros((boxes.shape[0], 3, 3))
        rot[:, 0, 0] = cos_yaw
        rot[:, 0, 1] = -sin_yaw
        rot[:, 1, 0] = sin_yaw
        rot[:, 1, 1] = cos_yaw
        rot[:, 2, 2] = 1.0
        corners = corners @ rot.transpose(1, 2)
        centers = torch.stack([x, y, z], dim=1)
        return corners + centers.unsqueeze(1)

    def _to_34(self, mat: torch.Tensor) -> torch.Tensor:
        if mat.shape[-2:] == (4, 4):
            return mat[:3, :4]
        if mat.shape[-2:] == (3, 4):
            return mat
        raise ValueError("calib matrix must have shape (4, 4) or (3, 4)")

    def _to_44(self, mat: torch.Tensor) -> torch.Tensor:
        if mat.shape[-2:] == (4, 4):
            return mat
        if mat.shape[-2:] == (3, 4):
            out = torch.eye(4, device=mat.device, dtype=mat.dtype)
            out[:3, :4] = mat
            return out
        if mat.shape[-2:] == (3, 3):
            out = torch.eye(4, device=mat.device, dtype=mat.dtype)
            out[:3, :3] = mat
            return out
        raise ValueError("calib matrix must have shape (4, 4), (3, 4), or (3, 3)")

    def _normalize_angle(self, value: float) -> float:
        return (value + math.pi) % (2.0 * math.pi) - math.pi

    def _project_bbox2d(
        self,
        boxes_lidar: torch.Tensor,
        batch_indices: torch.Tensor,
        batch: dict | None,
    ) -> torch.Tensor:
        if batch is None or boxes_lidar.numel() == 0:
            return boxes_lidar.new_zeros((boxes_lidar.shape[0], 4))

        calibs = batch.get("calibs", None)
        if calibs is None and all(k in batch for k in ("P2", "R0_rect", "Tr_velo2cam")):
            calibs = [
                {
                    "P2": batch["P2"],
                    "R0_rect": batch["R0_rect"],
                    "Tr_velo2cam": batch["Tr_velo2cam"],
                }
            ]
        if calibs is None:
            return boxes_lidar.new_zeros((boxes_lidar.shape[0], 4))

        image_shape = batch.get("image_shape", None)
        if image_shape is None and "images" in batch:
            image_shape = tuple(batch["images"].shape[-2:])

        corners_lidar = self._box_lidar_corners(boxes_lidar)
        bboxes = boxes_lidar.new_zeros((boxes_lidar.shape[0], 4))
        for idx, corners in enumerate(corners_lidar):
            batch_id = int(batch_indices[idx].item()) if batch_indices is not None else 0
            calib = calibs[min(batch_id, len(calibs) - 1)]
            p2 = self._to_34(calib["P2"]).to(device=boxes_lidar.device, dtype=boxes_lidar.dtype)
            r0 = self._to_44(calib["R0_rect"]).to(device=boxes_lidar.device, dtype=boxes_lidar.dtype)
            tr = self._to_44(calib["Tr_velo2cam"]).to(device=boxes_lidar.device, dtype=boxes_lidar.dtype)
            proj = p2 @ r0 @ tr

            ones = torch.ones((corners.shape[0], 1), device=boxes_lidar.device, dtype=boxes_lidar.dtype)
            corners_h = torch.cat([corners[:, :3], ones], dim=1)
            pixels_h = corners_h @ proj.t()
            depth = pixels_h[:, 2]
            valid = depth > 1e-6
            if not valid.any():
                continue
            pixels = pixels_h[valid, :2] / depth[valid, None].clamp_min(1e-6)
            x1, y1 = pixels.min(dim=0).values
            x2, y2 = pixels.max(dim=0).values
            if image_shape is not None:
                h, w = int(image_shape[0]), int(image_shape[1])
                x1 = x1.clamp(0, max(w - 1, 0))
                x2 = x2.clamp(0, max(w - 1, 0))
                y1 = y1.clamp(0, max(h - 1, 0))
                y2 = y2.clamp(0, max(h - 1, 0))
            bboxes[idx] = torch.stack([x1, y1, x2, y2])
        return bboxes

    def _lidar_to_camera_boxes(
        self,
        boxes_lidar: torch.Tensor,
        batch_indices: torch.Tensor,
        batch: dict | None,
    ) -> torch.Tensor:
        if batch is None or boxes_lidar.numel() == 0:
            return boxes_lidar

        calibs = batch.get("calibs", None)
        if calibs is None and all(k in batch for k in ("R0_rect", "Tr_velo2cam")):
            calibs = [
                {
                    "R0_rect": batch["R0_rect"],
                    "Tr_velo2cam": batch["Tr_velo2cam"],
                }
            ]
        if calibs is None:
            return boxes_lidar

        cam_boxes = boxes_lidar.clone()
        try:
            from mmdet3d.structures import Box3DMode
        except Exception:
            Box3DMode = None

        for batch_id in torch.unique(batch_indices):
            mask = batch_indices == batch_id
            idx = int(batch_id.item())
            calib = calibs[min(idx, len(calibs) - 1)]
            r0 = self._to_44(calib["R0_rect"]).to(device=boxes_lidar.device, dtype=boxes_lidar.dtype)
            tr = self._to_44(calib["Tr_velo2cam"]).to(device=boxes_lidar.device, dtype=boxes_lidar.dtype)
            rt = r0 @ tr
            if Box3DMode is not None:
                cam_boxes[mask] = Box3DMode.convert(
                    boxes_lidar[mask],
                    src=Box3DMode.LIDAR,
                    dst=Box3DMode.CAM,
                    rt_mat=rt,
                    correct_yaw=False,
                )
            else:
                centers_h = torch.cat(
                    [
                        boxes_lidar[mask, :3],
                        torch.ones((int(mask.sum()), 1), device=boxes_lidar.device, dtype=boxes_lidar.dtype),
                    ],
                    dim=1,
                )
                cam_centers = centers_h @ rt.t()
                cam_boxes[mask, :3] = cam_centers[:, :3]
                cam_boxes[mask, 3:6] = torch.stack(
                    [boxes_lidar[mask, 3], boxes_lidar[mask, 5], boxes_lidar[mask, 4]],
                    dim=1,
                )
                cam_boxes[mask, 6] = -boxes_lidar[mask, 6] - torch.pi / 2
        return cam_boxes

    def forward(self, detector_output: dict, batch: dict | None = None) -> list[str]:
        rois = detector_output.get("flat_rois", None)
        scores = detector_output.get("flat_roi_scores", None)
        labels = detector_output.get("flat_roi_labels", None)
        batch_indices = detector_output.get("flat_roi_batch_indices", None)

        if rois is None:
            rois_list = detector_output.get("rois", [])
            score_list = detector_output.get("roi_scores", [])
            label_list = detector_output.get("roi_labels", [])
            if len(rois_list) == 0:
                return []
            rois = rois_list[0]
            scores = score_list[0] if len(score_list) > 0 else torch.ones((rois.shape[0],), device=rois.device)
            labels = label_list[0] if len(label_list) > 0 else torch.zeros((rois.shape[0],), device=rois.device, dtype=torch.long)

        if scores is None:
            scores = torch.ones((rois.shape[0],), device=rois.device, dtype=rois.dtype)
        if labels is None:
            labels = torch.zeros((rois.shape[0],), device=rois.device, dtype=torch.long)
        if batch_indices is None:
            batch_indices = torch.zeros((rois.shape[0],), device=rois.device, dtype=torch.long)

        bboxes_2d = self._project_bbox2d(rois[:, :7], batch_indices, batch)
        cam_boxes = self._lidar_to_camera_boxes(rois[:, :7], batch_indices, batch)

        lines: list[str] = []
        for box, bbox2d, score, label in zip(
            cam_boxes.detach().cpu(),
            bboxes_2d.detach().cpu(),
            scores.detach().cpu(),
            labels.detach().cpu(),
        ):
            cls_id = int(label.item())
            name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else self.class_names[0]
            x, y, z, length, height, width, yaw = [float(v) for v in box[:7]]
            x1, y1, x2, y2 = [float(v) for v in bbox2d]
            score_v = float(score.item())
            alpha = self._normalize_angle(yaw - math.atan2(x, z))
            line = (
                f"{name} 0.00 0 {alpha:.4f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{height:.4f} {width:.4f} {length:.4f} "
                f"{x:.4f} {y:.4f} {z:.4f} {yaw:.4f} {score_v:.6f}"
            )
            lines.append(line)
        return lines
