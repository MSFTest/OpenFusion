# -*- coding: utf-8 -*-
"""
MatchAndPairModule

功能：
    几何匹配 + pair 特征构造

内部调用算子（已注册为 fusionrepo）：
    - compose_kitti_proj_matrix
    - points_to_image
    - iou2d_matrix
    - pair_selector
    - pair_feature_encoder

输入：
    boxes_2d:   (K,4)   xyxy
    scores_2d:  (K,)
    labels_2d:  (K,)
    boxes_3d:   (N,7)   LiDAR [x,y,z,w,l,h,yaw]
    scores_3d:  (N,)
    labels_3d:  (N,)
    calib: dict

输出：
    pair_features
    pair_indices
    K
    N
    boxes_3d / scores_3d / labels_3d   （注意：这里已经是 valid_box_mask 过滤后的 proposal）
"""

import torch
import torch.nn as nn


class MatchAndPairModule(nn.Module):
    def __init__(self, iou_threshold: float = 0.1):
        super().__init__()
        self.iou_threshold = float(iou_threshold)

    @staticmethod
    def _ensure_label_tensor(
        labels: torch.Tensor,
        num: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        labels 允许为 None；若为 None，则返回全 0，表示单类场景。
        """
        if labels is None:
            return torch.zeros((num,), dtype=torch.long, device=device)
        return labels.to(device=device, dtype=torch.long).view(-1)

    def forward(self,
                boxes_2d: torch.Tensor,
                scores_2d: torch.Tensor,
                labels_2d: torch.Tensor,
                boxes_3d: torch.Tensor,
                scores_3d: torch.Tensor,
                labels_3d: torch.Tensor,
                calib: dict):

        if boxes_3d.numel() > 0:
            device = boxes_3d.device
            dtype = boxes_3d.dtype
        elif boxes_2d.numel() > 0:
            device = boxes_2d.device
            dtype = boxes_2d.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        boxes_2d = boxes_2d.to(device=device, dtype=torch.float32).view(-1, 4)
        scores_2d = scores_2d.to(device=device, dtype=torch.float32).view(-1)
        boxes_3d = boxes_3d.to(device=device, dtype=torch.float32).view(-1, 7)
        scores_3d = scores_3d.to(device=device, dtype=torch.float32).view(-1)

        K = boxes_2d.shape[0]
        N = boxes_3d.shape[0]

        labels_2d = self._ensure_label_tensor(labels_2d, K, device)
        labels_3d = self._ensure_label_tensor(labels_3d, N, device)

        # 1) 构建投影矩阵
        R0_rect = calib["R0_rect"].to(device=device, dtype=dtype)
        Tr_velo_to_cam = calib["Tr_velo_to_cam"].to(device=device, dtype=dtype)
        P2 = calib["P2"].to(device=device, dtype=dtype)

        proj_matrix = torch.ops.fusionrepo.compose_kitti_proj_matrix(
            R0_rect,
            Tr_velo_to_cam,
            P2
        )

        # 2) 投影 3D 中心点（用于 pair_feature_encoder 中 distance / center 相关特征）
        if N == 0:
            pixel_coords = boxes_3d.new_zeros((0, 2))
        else:
            centers_3d = boxes_3d[:, 0:3]
            pixel_coords = torch.ops.fusionrepo.points_to_image(
                centers_3d,
                proj_matrix
            )

        # 3) 3D box -> 8 corners -> 2D bbox
        if N == 0:
            projected_boxes_2d = boxes_3d.new_zeros((0, 4))
            valid_box_mask = torch.zeros((0,), dtype=torch.bool, device=device)
        else:
            boxes = boxes_3d

            centers = boxes[:, 0:3]
            w = boxes[:, 3]
            l = boxes[:, 4]
            h = boxes[:, 5]
            yaw = boxes[:, 6]

            # LiDAR box corners
            x_c = torch.stack([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dim=1)
            y_c = torch.stack([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dim=1)
            z_c = torch.stack([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dim=1)

            corners = torch.stack([x_c, y_c, z_c], dim=2)  # (N,8,3)

            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)

            rot = torch.zeros((N, 3, 3), device=device, dtype=boxes.dtype)
            rot[:, 0, 0] = cos_yaw
            rot[:, 0, 1] = -sin_yaw
            rot[:, 1, 0] = sin_yaw
            rot[:, 1, 1] = cos_yaw
            rot[:, 2, 2] = 1.0

            corners = torch.bmm(corners, rot.transpose(1, 2))
            corners = corners + centers.unsqueeze(1)  # (N,8,3)

            # 构造成 4x4
            if R0_rect.shape == (3, 3):
                R0_4 = torch.eye(4, device=device, dtype=dtype)
                R0_4[:3, :3] = R0_rect
            else:
                R0_4 = R0_rect

            if Tr_velo_to_cam.shape == (3, 4):
                Tr_4 = torch.eye(4, device=device, dtype=dtype)
                Tr_4[:3, :4] = Tr_velo_to_cam
            else:
                Tr_4 = Tr_velo_to_cam

            corners_flat = corners.reshape(-1, 3)
            ones = torch.ones((corners_flat.shape[0], 1), device=device, dtype=dtype)
            corners_h = torch.cat([corners_flat, ones], dim=1)  # (N*8,4)

            pts_cam = corners_h @ Tr_4.t()
            pts_rect = pts_cam @ R0_4.t()

            depth = pts_rect[:, 2].reshape(N, 8)
            valid_corner_mask = depth > 1e-3

            pixels_h = pts_rect[:, :4] @ P2.t()
            uv_all = pixels_h[:, :2] / pixels_h[:, 2:3].clamp_min(1e-6)
            uv_all = uv_all.reshape(N, 8, 2)

            valid_box_mask = valid_corner_mask.any(dim=1)

            xmin = boxes.new_full((N,), 0.0)
            ymin = boxes.new_full((N,), 0.0)
            xmax = boxes.new_full((N,), 0.0)
            ymax = boxes.new_full((N,), 0.0)

            if valid_box_mask.any():
                uv_x = uv_all[:, :, 0]
                uv_y = uv_all[:, :, 1]

                inf = torch.tensor(float("inf"), device=device, dtype=dtype)
                ninf = torch.tensor(float("-inf"), device=device, dtype=dtype)

                uv_x_min = torch.where(valid_corner_mask, uv_x, inf)
                uv_y_min = torch.where(valid_corner_mask, uv_y, inf)
                uv_x_max = torch.where(valid_corner_mask, uv_x, ninf)
                uv_y_max = torch.where(valid_corner_mask, uv_y, ninf)

                xmin = uv_x_min.min(dim=1).values
                ymin = uv_y_min.min(dim=1).values
                xmax = uv_x_max.max(dim=1).values
                ymax = uv_y_max.max(dim=1).values

            projected_boxes_2d = torch.stack([xmin, ymin, xmax, ymax], dim=1)

            # 只保留合法前向框
            valid_box_mask = valid_box_mask & (xmax > xmin) & (ymax > ymin)

            projected_boxes_2d = projected_boxes_2d[valid_box_mask]
            boxes_3d = boxes_3d[valid_box_mask]
            scores_3d = scores_3d[valid_box_mask]
            labels_3d = labels_3d[valid_box_mask]
            pixel_coords = pixel_coords[valid_box_mask]

            N = boxes_3d.shape[0]

        # 4) IoU 计算
        if K == 0 or N == 0:
            iou_matrix = boxes_2d.new_zeros((K, N))
        else:
            iou_matrix = torch.ops.fusionrepo.iou2d_matrix(
                boxes_2d,
                projected_boxes_2d
            )

            # 论文对齐：只在同类别之间做匹配
            same_class = (labels_2d.view(K, 1) == labels_3d.view(1, N))
            iou_matrix = iou_matrix * same_class.to(iou_matrix.dtype)

        # 5) pair selection
        pair_indices = torch.ops.fusionrepo.pair_selector(
            iou_matrix,
            self.iou_threshold
        )

        # 6) pair features
        pair_features = torch.ops.fusionrepo.pair_feature_encoder(
            pair_indices,
            iou_matrix,
            boxes_3d,
            scores_2d,
            scores_3d
        )

        return {
            "pair_features": pair_features,
            "pair_indices": pair_indices,
            "K": K,
            "N": N,
            "boxes_3d": boxes_3d,
            "scores_3d": scores_3d,
            "labels_3d": labels_3d,
            "projected_boxes_2d": projected_boxes_2d,
            "pixel_coords": pixel_coords
        }