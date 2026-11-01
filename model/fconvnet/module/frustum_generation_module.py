# -*- coding: utf-8 -*-
"""
FrustumGenerationModule

功能：
    根据 2D 检测框与点云生成 F-ConvNet 的 sliding frustum 点组。
    本模块组合投影、2D 框筛选、深度分桶和按 frustum 分组算子。
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


class FrustumGenerationModule(nn.Module):
    """
    Sliding frustum 生成模块。

    Args:
        bin_size: 深度 bin 高度。
        stride: 深度 bin 滑动步长。
        max_points_per_frustum: 每个 frustum 最多保留点数。
        image_width: 默认图像宽度。
        image_height: 默认图像高度。

    Inputs:
        points: (B, N, 3) 或 (N, 3)，相机坐标系点云。
        det2d: dict，包含 boxes_2d / scores_2d / labels_2d。
        cam_intrinsic: (B, 3, 3) 或 (3, 3)。
        image_shape: (H, W) 或 None。

    Returns:
        dict:
            frustum_points: (P, T, M, 3)
            frustum_mask: (P, T, M)
            frustum_bins: (P, T, 2)
            center_refs: (P, T, 3)
            proposal_boxes_2d: (P, 4)
            proposal_scores: (P,)
            proposal_labels: (P,)
            proposal_batch_indices: (P,)
    """

    def __init__(
        self,
        bin_size: float = 0.25,
        stride: float = 0.25,
        max_depth: float = 70.0,
        max_points_per_frustum: int = 128,
        image_width: int = 1242,
        image_height: int = 375,
    ):
        super().__init__()
        self.bin_size = float(bin_size)
        self.stride = float(stride)
        self.max_depth = float(max_depth)
        self.max_points_per_frustum = int(max_points_per_frustum)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.scale_strides = (0.25, 0.5, 1.0, 2.0)
        self.scale_radii = (0.25, 0.5, 1.0, 2.0)
        self.scale_samples = (32, 64, 64, max(128, self.max_points_per_frustum))

    @staticmethod
    def _rotate_y(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        out = points.clone()
        out[..., 0] = c * points[..., 0] - s * points[..., 2]
        out[..., 2] = s * points[..., 0] + c * points[..., 2]
        return out

    @staticmethod
    def _group_at_references(
        points: torch.Tensor,
        refs: torch.Tensor,
        radius: float,
        nsample: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match the reference CUDA op: first ``nsample`` points with |dz| < radius."""
        n = points.shape[0]
        t = refs.shape[0]
        if n == 0:
            return points.new_zeros((t, nsample, 3)), torch.zeros(
                (t, nsample), device=points.device, dtype=torch.bool
            )
        valid = (points[:, 2].unsqueeze(0) - refs[:, 2].unsqueeze(1)).abs() < radius
        source_idx = torch.arange(n, device=points.device).view(1, n).expand(t, n)
        ranked = source_idx.masked_fill(~valid, n)
        k = min(nsample, n)
        selected = torch.topk(ranked, k=k, dim=1, largest=False, sorted=True).values
        selected_valid = selected < n
        selected = selected.clamp_max(n - 1)
        grouped = points[selected]
        if k < nsample:
            grouped = torch.cat([grouped, grouped.new_zeros((t, nsample - k, 3))], dim=1)
            selected_valid = torch.cat(
                [selected_valid, torch.zeros((t, nsample - k), device=points.device, dtype=torch.bool)],
                dim=1,
            )
        grouped = grouped.masked_fill(~selected_valid.unsqueeze(-1), 0)
        return grouped, selected_valid

    @staticmethod
    def _as_batched_points(points: torch.Tensor) -> torch.Tensor:
        if points.dim() == 2:
            return points.unsqueeze(0)
        if points.dim() == 3:
            return points
        raise ValueError("points must have shape (N,3) or (B,N,3)")

    @staticmethod
    def _select_batch_tensor(x: torch.Tensor, b: int) -> torch.Tensor:
        if x.dim() >= 3:
            return x[b]
        return x

    def forward(
        self,
        points: torch.Tensor,
        det2d: Dict[str, torch.Tensor],
        cam_intrinsic: torch.Tensor,
        image_shape: tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        points_b = self._as_batched_points(points)
        B = points_b.shape[0]
        H, W = image_shape if image_shape is not None else (self.image_height, self.image_width)

        boxes_2d = det2d["boxes_2d"]
        scores_2d = det2d.get("scores_2d")
        labels_2d = det2d.get("labels_2d")
        batch_indices = det2d.get("batch_indices")
        proposal_gt_indices = det2d.get("proposal_gt_indices")

        if boxes_2d.dim() == 2:
            if batch_indices is None:
                batch_indices = torch.zeros((boxes_2d.shape[0],), device=boxes_2d.device, dtype=torch.long)
        elif boxes_2d.dim() == 3:
            K = boxes_2d.shape[1]
            batch_indices = torch.arange(B, device=boxes_2d.device).view(B, 1).expand(B, K).reshape(-1)
            boxes_2d = boxes_2d.reshape(B * K, 4)
            if scores_2d is not None:
                scores_2d = scores_2d.reshape(B * K)
            if labels_2d is not None:
                labels_2d = labels_2d.reshape(B * K)
        else:
            raise ValueError("boxes_2d must have shape (K,4) or (B,K,4)")

        if scores_2d is None:
            scores_2d = torch.ones((boxes_2d.shape[0],), device=boxes_2d.device, dtype=boxes_2d.dtype)
        if labels_2d is None:
            labels_2d = torch.zeros((boxes_2d.shape[0],), device=boxes_2d.device, dtype=torch.long)

        multiscale_points_list: List[List[torch.Tensor]] = [[] for _ in self.scale_strides]
        multiscale_mask_list: List[List[torch.Tensor]] = [[] for _ in self.scale_strides]
        multiscale_refs_list: List[List[torch.Tensor]] = [[] for _ in self.scale_strides]
        camera_refs_list: List[torch.Tensor] = []
        rot_angles: List[torch.Tensor] = []
        kept_boxes: List[torch.Tensor] = []
        kept_scores: List[torch.Tensor] = []
        kept_labels: List[torch.Tensor] = []
        kept_batches: List[torch.Tensor] = []
        kept_gt_indices: List[torch.Tensor] = []

        for i in range(boxes_2d.shape[0]):
            b = int(batch_indices[i].item())
            pts = points_b[b]
            K_b = self._select_batch_tensor(cam_intrinsic, b)
            uv, depth = torch.ops.fusionrepo.frustum_points_to_image(pts, K_b, 1e-6)
            in_box = torch.ops.fusionrepo.points_in_2d_boxes_mask(
                uv, boxes_2d[i], int(W), int(H), True
            )
            if not bool(in_box.any()):
                continue

            pts_sel = pts[in_box]
            box_center = (boxes_2d[i, :2] + boxes_2d[i, 2:]) / 2
            unit_pixel = torch.stack([box_center[0], box_center[1], box_center.new_tensor(1.0)])
            unit_ray = torch.linalg.solve(K_b, unit_pixel)
            rot_angle = torch.atan2(unit_ray[0], unit_ray[2])
            pts_center = self._rotate_y(pts_sel, rot_angle)
            per_scale_camera_refs = []
            for scale_idx, (scale_stride, radius, nsample) in enumerate(
                zip(self.scale_strides, self.scale_radii, self.scale_samples)
            ):
                depth_centers = torch.arange(
                    scale_stride / 2,
                    self.max_depth,
                    scale_stride,
                    device=pts_sel.device,
                    dtype=pts_sel.dtype,
                )
                pixels_at_depth = torch.stack(
                    [box_center[0] * depth_centers, box_center[1] * depth_centers, depth_centers],
                    dim=1,
                )
                refs_camera = torch.linalg.solve(K_b, pixels_at_depth.transpose(0, 1)).transpose(0, 1)
                refs_center = self._rotate_y(refs_camera, rot_angle)
                grouped_points, grouped_mask = self._group_at_references(
                    pts_center, refs_center, radius, nsample
                )
                multiscale_points_list[scale_idx].append(grouped_points)
                multiscale_mask_list[scale_idx].append(grouped_mask)
                multiscale_refs_list[scale_idx].append(refs_center)
                per_scale_camera_refs.append(refs_camera)
            camera_refs_list.append(per_scale_camera_refs[1])
            rot_angles.append(rot_angle)
            kept_boxes.append(boxes_2d[i])
            kept_scores.append(scores_2d[i])
            kept_labels.append(labels_2d[i])
            kept_batches.append(batch_indices[i])
            if proposal_gt_indices is not None:
                kept_gt_indices.append(proposal_gt_indices[i])

        if len(kept_boxes) == 0:
            device = points.device
            dtype = points.dtype
            lengths = [int(self.max_depth / stride) for stride in self.scale_strides]
            multi_points = [
                torch.zeros((0, length, nsample, 3), device=device, dtype=dtype)
                for length, nsample in zip(lengths, self.scale_samples)
            ]
            multi_masks = [
                torch.zeros((0, length, nsample), device=device, dtype=torch.bool)
                for length, nsample in zip(lengths, self.scale_samples)
            ]
            multi_refs = [torch.zeros((0, length, 3), device=device, dtype=dtype) for length in lengths]
            result = {
                "multiscale_points": multi_points,
                "multiscale_masks": multi_masks,
                "multiscale_center_refs": multi_refs,
                "frustum_points": multi_points[0],
                "frustum_mask": multi_masks[0],
                "frustum_bins": torch.zeros((0, lengths[0], 2), device=device, dtype=dtype),
                "center_refs": multi_refs[1],
                "camera_center_refs": multi_refs[1],
                "frustum_rot_angles": torch.zeros((0,), device=device, dtype=dtype),
                "proposal_boxes_2d": torch.zeros((0, 4), device=device, dtype=dtype),
                "proposal_scores": torch.zeros((0,), device=device, dtype=dtype),
                "proposal_labels": torch.zeros((0,), device=device, dtype=torch.long),
                "proposal_batch_indices": torch.zeros((0,), device=device, dtype=torch.long),
            }
            if proposal_gt_indices is not None:
                result["proposal_gt_indices"] = torch.zeros((0,), device=device, dtype=torch.long)
            return result

        multi_points = [torch.stack(items, dim=0) for items in multiscale_points_list]
        multi_masks = [torch.stack(items, dim=0) for items in multiscale_mask_list]
        multi_refs = [torch.stack(items, dim=0) for items in multiscale_refs_list]
        fine_depth = torch.arange(
            self.scale_strides[0] / 2, self.max_depth, self.scale_strides[0],
            device=points.device, dtype=points.dtype,
        )
        out_bins = torch.stack(
            [fine_depth - self.scale_radii[0], fine_depth + self.scale_radii[0]], dim=1
        ).unsqueeze(0).expand(len(kept_boxes), -1, -1)

        result = {
            "multiscale_points": multi_points,
            "multiscale_masks": multi_masks,
            "multiscale_center_refs": multi_refs,
            "frustum_points": multi_points[0],
            "frustum_mask": multi_masks[0],
            "frustum_bins": out_bins,
            "center_refs": multi_refs[1],
            "camera_center_refs": torch.stack(camera_refs_list, dim=0),
            "frustum_rot_angles": torch.stack(rot_angles),
            "proposal_boxes_2d": torch.stack(kept_boxes, dim=0),
            "proposal_scores": torch.stack(kept_scores, dim=0),
            "proposal_labels": torch.stack(kept_labels, dim=0).long(),
            "proposal_batch_indices": torch.stack(kept_batches, dim=0).long(),
        }
        if kept_gt_indices:
            result["proposal_gt_indices"] = torch.stack(kept_gt_indices).long()
        return result
