from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .fconvnet_detection_head import FConvNetDetectionHead
from .frustum_fcn_module import FrustumFCNModule
from .frustum_generation_module import FrustumGenerationModule
from .frustum_pointnet_encoding_module import FrustumPointNetEncodingModule


class FConvNetRefinementModule(nn.Module):
    """Second F-ConvNet stage operating in each predicted box's local frame."""

    def __init__(
        self,
        max_points_per_group: int = 128,
        residual_scale: float = 1.0,
        residual_gate_score: float = float("inf"),
    ):
        super().__init__()
        self.strides = (0.1, 0.2, 0.4, 0.8)
        self.radii = self.strides
        self.samples = (32, 64, 64, max(128, int(max_points_per_group)))
        self.residual_scale = float(residual_scale)
        self.residual_gate_score = float(residual_gate_score)
        self.pointnet = FrustumPointNetEncodingModule(in_channels=3)
        self.fcn = FrustumFCNModule(in_channels=128, num_classes=3)
        self.head = FConvNetDetectionHead(in_channels=self.fcn.out_channels)
        nn.init.zeros_(self.head.reg_out.weight)
        nn.init.zeros_(self.head.reg_out.bias)

    @staticmethod
    def camera_to_local(
        points: torch.Tensor, base_centers: torch.Tensor, base_yaw: torch.Tensor
    ) -> torch.Tensor:
        delta = points - base_centers
        c, s = torch.cos(base_yaw), torch.sin(base_yaw)
        out = delta.clone()
        out[..., 0] = c * delta[..., 0] - s * delta[..., 2]
        out[..., 2] = s * delta[..., 0] + c * delta[..., 2]
        return out

    @staticmethod
    def local_to_camera(
        points: torch.Tensor, base_centers: torch.Tensor, base_yaw: torch.Tensor
    ) -> torch.Tensor:
        c, s = torch.cos(base_yaw), torch.sin(base_yaw)
        out = points.clone()
        out[..., 0] = c * points[..., 0] + s * points[..., 2]
        out[..., 2] = -s * points[..., 0] + c * points[..., 2]
        return out + base_centers

    @staticmethod
    def select_base_boxes(stage1: Dict[str, torch.Tensor]) -> torch.Tensor:
        foreground = torch.softmax(stage1["cls_logits"], dim=-1)[..., 1]
        best = foreground.argmax(dim=1)
        rows = torch.arange(best.shape[0], device=best.device)
        return stage1["boxes_3d"][rows, best].detach()

    @staticmethod
    def jitter_gt_base_boxes(
        gt_boxes: torch.Tensor,
        proposal_batches: torch.Tensor,
        proposal_gt_indices: torch.Tensor,
        ratio: float = 0.05,
    ) -> torch.Tensor:
        matched = gt_boxes[proposal_batches.long(), proposal_gt_indices.long()]
        center = matched[:, :3].clone()
        center[:, 1] -= matched[:, 5] / 2
        center = center + (torch.rand_like(center) * 2 - 1) * matched[:, [4, 5, 3]] * ratio
        size = matched[:, 3:6] * (1 + (torch.rand_like(matched[:, 3:6]) * 2 - 1) * ratio)
        yaw = matched[:, 6:7] + (torch.rand_like(matched[:, 6:7]) * 2 - 1) * ratio * torch.pi
        return torch.cat([center, size, yaw], dim=1)

    def _references_and_valid(
        self, base_boxes: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        refs_all, valid_all = [], []
        lengths = base_boxes[:, 3].clamp_min(self.strides[0])
        for stride in self.strides:
            counts = torch.ceil(lengths / stride).long().clamp_min(1)
            max_count = int(counts.max().item())
            index = torch.arange(max_count, device=base_boxes.device)
            z = -lengths[:, None] / 2 + (index.to(base_boxes.dtype)[None] + 0.5) * stride
            valid = index[None] < counts[:, None]
            refs = base_boxes.new_zeros((base_boxes.shape[0], max_count, 3))
            refs[..., 2] = z
            refs_all.append(refs)
            valid_all.append(valid)
        return refs_all, valid_all

    def generate_local_references(self, base_boxes: torch.Tensor) -> list[torch.Tensor]:
        return self._references_and_valid(base_boxes)[0]

    def decode_residual_boxes(
        self, raw: torch.Tensor, base_boxes: torch.Tensor
    ) -> torch.Tensor:
        bins = self.head.num_heading_bins
        center = raw[..., :3] * self.residual_scale
        heading_scores = raw[..., 3:3 + bins]
        heading_residuals = raw[..., 3 + bins:3 + 2 * bins]
        heading_class = heading_scores.argmax(dim=-1)
        selected_heading = heading_residuals.gather(
            2, heading_class.unsqueeze(-1)
        ).squeeze(-1)
        angle_per_bin = 2.0 * torch.pi / bins
        heading = heading_class.to(raw.dtype) * angle_per_bin + selected_heading * (
            angle_per_bin / 2
        )
        heading = (
            torch.remainder(heading + torch.pi, 2 * torch.pi) - torch.pi
        ) * self.residual_scale
        size_start = 3 + 2 * bins
        size_residual = raw[..., size_start + 1:size_start + 4]
        size = (
            base_boxes[:, None, 3:6] * (1.0 + size_residual * self.residual_scale)
        ).clamp_min(1e-3)
        return torch.cat([center, size, heading.unsqueeze(-1)], dim=-1)

    def apply_confidence_gate(
        self,
        local_boxes: torch.Tensor,
        base_boxes: torch.Tensor,
        combined_scores: torch.Tensor,
    ) -> torch.Tensor:
        apply = (combined_scores < self.residual_gate_score).to(local_boxes)
        apply = apply[:, None, None]
        base_local = local_boxes.new_zeros(local_boxes.shape)
        base_local[..., 3:6] = base_boxes[:, None, 3:6]
        return base_local + apply * (local_boxes - base_local)

    def forward(
        self, stage1: Dict[str, torch.Tensor], points: torch.Tensor,
        gt_boxes: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        base_boxes = self.select_base_boxes(stage1)
        if self.training and gt_boxes is not None and "proposal_gt_indices" in stage1:
            base_boxes = self.jitter_gt_base_boxes(
                gt_boxes, stage1["proposal_batch_indices"], stage1["proposal_gt_indices"]
            )
        if base_boxes.shape[0] == 0:
            return stage1
        base_centers = base_boxes[:, :3]
        base_yaw = base_boxes[:, 6]
        stage1_foreground = torch.softmax(stage1["cls_logits"], dim=-1)[..., 1]
        best_stage1 = stage1_foreground.argmax(dim=1)
        rows = torch.arange(best_stage1.shape[0], device=best_stage1.device)
        base_foreground_scores = stage1_foreground[rows, best_stage1].detach()
        batch_indices = stage1["proposal_batch_indices"]
        refs, ref_valid = self._references_and_valid(base_boxes)
        grouped_scales: list[list[torch.Tensor]] = [[] for _ in self.strides]
        masks_scales: list[list[torch.Tensor]] = [[] for _ in self.strides]

        for proposal_idx in range(base_boxes.shape[0]):
            batch_idx = int(batch_indices[proposal_idx].item())
            local = self.camera_to_local(
                points[batch_idx], base_centers[proposal_idx], base_yaw[proposal_idx]
            )
            half_extent = base_boxes[proposal_idx, [4, 5, 3]] * 0.6
            crop = (local.abs() <= half_extent).all(dim=1)
            local = local[crop]
            for scale_idx, (radius, nsample) in enumerate(zip(self.radii, self.samples)):
                grouped, mask = FrustumGenerationModule._group_at_references(
                    local, refs[scale_idx][proposal_idx], radius, nsample
                )
                mask &= ref_valid[scale_idx][proposal_idx, :, None]
                grouped_scales[scale_idx].append(grouped)
                masks_scales[scale_idx].append(mask)

        grouped = [torch.stack(items) for items in grouped_scales]
        masks = [torch.stack(items) for items in masks_scales]
        features = self.pointnet(grouped, masks, refs)
        fused = self.fcn(features, stage1["proposal_labels"])
        output_refs = refs[1][:, : fused.shape[-1]]
        pred = self.head(fused, output_refs)
        local_boxes = self.decode_residual_boxes(pred["reg_raw"], base_boxes)
        combined_scores = base_foreground_scores + stage1["proposal_scores"].to(
            base_foreground_scores
        )
        local_boxes = self.apply_confidence_gate(
            local_boxes, base_boxes, combined_scores
        )
        camera_boxes = local_boxes.clone()
        camera_boxes[..., :3] = self.local_to_camera(
            local_boxes[..., :3], base_centers[:, None, :], base_yaw[:, None]
        )
        camera_boxes[..., 6] = torch.remainder(
            local_boxes[..., 6] + base_yaw[:, None] + torch.pi, 2 * torch.pi
        ) - torch.pi
        camera_refs = self.local_to_camera(
            output_refs, base_centers[:, None, :], base_yaw[:, None]
        )
        pred.update({
            "boxes_3d_center_view": local_boxes,
            "boxes_3d": camera_boxes,
            "center_refs": output_refs,
            "camera_center_refs": camera_refs,
            "candidate_valid_mask": ref_valid[1][:, : fused.shape[-1]],
            "frustum_rot_angles": base_yaw,
            "refine_base_centers": base_centers,
            "refine_base_boxes": base_boxes,
            "refine_residual": True,
            "base_foreground_scores": base_foreground_scores,
            "proposal_boxes_2d": stage1["proposal_boxes_2d"],
            "proposal_scores": stage1["proposal_scores"],
            "proposal_labels": stage1["proposal_labels"],
            "proposal_batch_indices": batch_indices,
            "stage1_pred": stage1,
            "lidar_out": stage1["lidar_out"],
        })
        if "proposal_gt_indices" in stage1:
            pred["proposal_gt_indices"] = stage1["proposal_gt_indices"]
        return pred
