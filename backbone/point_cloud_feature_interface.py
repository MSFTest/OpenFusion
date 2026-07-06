# -*- coding: utf-8 -*-
"""
PointCloudFeatureInterface

纯点云特征提取接口，正确处理 mmdet3d VoxelNet/PointPillars 的数据流。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Optional, List, Tuple


class PointCloudFeatureInterface(nn.Module):
    """
    点云特征提取接口

    输入:
        points: list[Tensor] 或 Tensor

    输出:
        dict with keys:
            - pts_feats: (N, C) 体素特征
            - voxel_coords: (N, 4) [batch_idx, z, y, x]
            - voxels: (N, T, 3) 体素内点坐标
            - voxel_num_points: (N,) 每个体素内点数
            - flat_rois: (K, 7) 检测框
            - flat_roi_batch_indices: (K,) batch 索引

    参数:
        extractor: 点云模型（VoxelNet/PointPillars）
        feat_index: 若返回多层特征，选择哪一层
    """

    def __init__(
        self,
        extractor: nn.Module,
        feat_index: Optional[int] = None,
        use_detection: bool = True,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        point_cloud_range: Optional[Tuple[float, float, float, float, float, float]] = None
    ):
        super().__init__()
        self.extractor = extractor
        self.feat_index = feat_index
        self.use_detection = use_detection
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def _cfg_get(self, obj: Any, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key, None)
        if hasattr(obj, "get"):
            try:
                return obj.get(key, None)
            except Exception:
                pass
        return getattr(obj, key, None)

    def _infer_voxel_params(self):
        voxel_size = self.voxel_size
        point_cloud_range = self.point_cloud_range

        cfg = getattr(self.extractor, "cfg", None)
        if cfg is not None:
            data_preprocessor = self._cfg_get(cfg, "data_preprocessor")
            voxel_layer = self._cfg_get(data_preprocessor, "voxel_layer")
            if voxel_size is None:
                voxel_size = self._cfg_get(voxel_layer, "voxel_size")
            if point_cloud_range is None:
                point_cloud_range = self._cfg_get(voxel_layer, "point_cloud_range")

        if voxel_size is None:
            voxel_layer = getattr(getattr(self.extractor, "data_preprocessor", None), "voxel_layer", None)
            voxel_size = getattr(voxel_layer, "voxel_size", None)
            point_cloud_range = point_cloud_range or getattr(voxel_layer, "point_cloud_range", None)

        if voxel_size is None or point_cloud_range is None:
            return None, None

        voxel_size = tuple(float(x) for x in voxel_size)
        point_cloud_range = tuple(float(x) for x in point_cloud_range)
        return voxel_size, point_cloud_range

    def _compute_voxel_centers_lidar(self, voxel_coords: torch.Tensor):
        voxel_size, point_cloud_range = self._infer_voxel_params()
        if voxel_size is None or point_cloud_range is None or voxel_coords is None:
            return None
        if voxel_coords.dim() != 2 or voxel_coords.shape[1] not in (3, 4):
            return None

        device = voxel_coords.device
        dtype = torch.float32
        voxel_size_t = torch.tensor(voxel_size, device=device, dtype=dtype)
        pc_min_t = torch.tensor(point_cloud_range[:3], device=device, dtype=dtype)

        coords = voxel_coords[:, 1:4] if voxel_coords.shape[1] == 4 else voxel_coords
        coords = coords.to(dtype)
        # mmdet3d voxel coordinates are [batch, z, y, x] or [z, y, x].
        xyz_indices = torch.stack([coords[:, 2], coords[:, 1], coords[:, 0]], dim=1)
        return (xyz_indices + 0.5) * voxel_size_t + pc_min_t

    def _feature_layout(self, feat: torch.Tensor) -> str:
        if not torch.is_tensor(feat):
            return "unknown"
        if feat.dim() == 2:
            return "token_nc"
        if feat.dim() == 4:
            # Most mmdet3d BEV features are (B, C, H, W). Keep this as metadata
            # only; legacy pts_feats flattening below remains unchanged.
            if feat.shape[1] <= 2048 and feat.shape[1] >= feat.shape[-1]:
                return "bev_bchw"
            return "bev_bhwc"
        if hasattr(feat, "features") and hasattr(feat, "indices"):
            return "sparse_tensor"
        return f"tensor_{feat.dim()}d"

    def _normalize_points_input(
        self,
        points
    ) -> List[torch.Tensor]:
        """统一为 list[Tensor]"""
        if torch.is_tensor(points):
            return [points]
        if isinstance(points, (list, tuple)):
            return list(points)
        raise TypeError("points must be Tensor or list/tuple of Tensor")

    def _build_data_samples(self, batch_size: int) -> List:
        """构造 data_samples"""
        from mmdet3d.structures import Box3DMode, Det3DDataSample, LiDARInstance3DBoxes

        samples = []
        for _ in range(batch_size):
            ds = Det3DDataSample()
            ds.set_metainfo({
                "box_type_3d": LiDARInstance3DBoxes,
                "box_mode_3d": Box3DMode.LIDAR
            })
            samples.append(ds)
        return samples

    def forward(self, points, **kwargs) -> dict:
        """
        Args:
            points: 点云数据
            **kwargs: 额外参数会被忽略

        Returns:
            dict with pts_feats, voxel_coords, voxels, voxel_num_points, flat_rois, flat_roi_batch_indices
        """
        points_list = self._normalize_points_input(points)
        batch_size = len(points_list)

        # 构建 detector 输入格式
        batch_inputs_dict = {"points": points_list}
        batch_data_samples = self._build_data_samples(batch_size)

        # 调用 data_preprocessor 进行体素化
        if hasattr(self.extractor, "data_preprocessor"):
            data = {
                "inputs": batch_inputs_dict,
                "data_samples": batch_data_samples
            }
            processed = self.extractor.data_preprocessor(data, training=self.training)
            batch_inputs_dict = processed["inputs"]
            batch_data_samples = processed.get("data_samples", batch_data_samples)

        # 提取体素信息
        voxels = None
        voxel_num_points = None
        voxel_coords = None

        if "voxels" in batch_inputs_dict:
            voxel_dict = batch_inputs_dict["voxels"]
            if isinstance(voxel_dict, dict):
                voxels = voxel_dict.get("voxels", None)
                voxel_coords = voxel_dict.get("coors", None)
                voxel_num_points = voxel_dict.get("num_points", None)
            elif isinstance(voxel_dict, (list, tuple)) and len(voxel_dict) >= 3:
                voxels = voxel_dict[0]
                voxel_num_points = voxel_dict[1]
                voxel_coords = voxel_dict[2]

        # 处理 voxels 格式
        if voxels is not None and voxels.dim() == 3 and voxels.shape[-1] >= 3:
            voxels = voxels[..., :3].contiguous()

        aligned_voxel_features = None
        if (
            voxels is not None
            and voxel_coords is not None
            and voxel_num_points is not None
            and hasattr(self.extractor, "voxel_encoder")
        ):
            try:
                voxel_input = batch_inputs_dict["voxels"]
                aligned_voxel_features = self.extractor.voxel_encoder(
                    voxel_input["voxels"],
                    voxel_input["num_points"],
                    voxel_input["coors"]
                )
            except Exception:
                aligned_voxel_features = None

        # 提取特征 - 传入 data_preprocessor 处理后的 batch_inputs_dict
        feats = self.extractor.extract_feat(batch_inputs_dict)

        if aligned_voxel_features is not None:
            pts_feats = aligned_voxel_features
        elif isinstance(feats, dict):
            candidate_keys = (
                "voxel_features",
                "fusion_keypoint_features",
                "keypoint_features",
                "spatial_features",
                "spatial_feats",
                "neck_feats",
            )
            pts_feats = None
            for key in candidate_keys:
                value = feats.get(key, None)
                if torch.is_tensor(value):
                    pts_feats = value
                    break
                if isinstance(value, (list, tuple)) and len(value) > 0 and torch.is_tensor(value[0]):
                    pts_feats = value[0]
                    break
            if pts_feats is None:
                raise ValueError(f"Cannot select tensor feature from extract_feat dict keys={list(feats.keys())}")
        elif isinstance(feats, (list, tuple)):
            if self.feat_index is not None:
                pts_feats = feats[self.feat_index]
            else:
                pts_feats = feats[0] if len(feats) > 0 else feats
        else:
            pts_feats = feats

        selected_feat = pts_feats
        feature_layout = self._feature_layout(selected_feat)

        # 如果是4D特征图 (B, H, W, C)，展平为2D (B*H*W, C)
        if pts_feats.dim() == 4:
            B, H, W, C = pts_feats.shape
            pts_feats = pts_feats.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            pts_feats = pts_feats.view(B, C, H * W)  # (B, C, H*W)
            pts_feats = pts_feats.permute(0, 2, 1).contiguous()  # (B, H*W, C)
            pts_feats = pts_feats.view(-1, C)  # (B*H*W, C)

        voxel_centers_lidar = self._compute_voxel_centers_lidar(voxel_coords)

        result = {
            "pts_feats": pts_feats,
            "voxel_features": aligned_voxel_features,
            "raw_feats": feats,
            "selected_feat": selected_feat,
            "feature_layout": feature_layout,
            "voxel_coords": voxel_coords,
            "voxel_centers_lidar": voxel_centers_lidar,
            "voxels": voxels,
            "voxel_num_points": voxel_num_points,
            "batch_inputs_dict": batch_inputs_dict,
            "batch_data_samples": batch_data_samples,
        }

        # 获取检测框
        if self.use_detection:
            try:
                # 使用 torch.enable_grad 来确保梯度流可以继续
                with torch.enable_grad():
                    preds = self.extractor.predict(
                        batch_inputs_dict,
                        batch_data_samples
                    )

                # 解析检测框
                flat_rois_list = []
                flat_batch_idx_list = []
                flat_score_list = []
                flat_label_list = []

                for b, pred in enumerate(preds):
                    if hasattr(pred, "pred_instances_3d"):
                        inst = pred.pred_instances_3d
                        boxes = inst.bboxes_3d
                        if hasattr(boxes, "tensor"):
                            boxes = boxes.tensor
                        if boxes.numel() > 0:
                            flat_rois_list.append(boxes)
                            flat_batch_idx_list.append(
                                torch.full((boxes.shape[0],), b,
                                          dtype=torch.long, device=boxes.device)
                            )
                            scores = getattr(inst, "scores_3d", None)
                            if scores is None:
                                scores = getattr(inst, "scores", None)
                            if scores is not None:
                                flat_score_list.append(scores.to(device=boxes.device))
                            labels = getattr(inst, "labels_3d", None)
                            if labels is None:
                                labels = getattr(inst, "labels", None)
                            if labels is not None:
                                flat_label_list.append(labels.to(device=boxes.device, dtype=torch.long))

                if len(flat_rois_list) > 0:
                    result["flat_rois"] = torch.cat(flat_rois_list, dim=0)
                    result["flat_roi_batch_indices"] = torch.cat(flat_batch_idx_list, dim=0)
                    if len(flat_score_list) == len(flat_rois_list):
                        result["flat_roi_scores"] = torch.cat(flat_score_list, dim=0)
                    else:
                        result["flat_roi_scores"] = torch.ones(
                            (result["flat_rois"].shape[0],),
                            dtype=result["flat_rois"].dtype,
                            device=result["flat_rois"].device
                        )
                    if len(flat_label_list) == len(flat_rois_list):
                        result["flat_roi_labels"] = torch.cat(flat_label_list, dim=0)
                    else:
                        result["flat_roi_labels"] = torch.zeros(
                            (result["flat_rois"].shape[0],),
                            dtype=torch.long,
                            device=result["flat_rois"].device
                        )
                else:
                    device = pts_feats.device
                    result["flat_rois"] = pts_feats.new_zeros((0, 7))
                    result["flat_roi_batch_indices"] = torch.zeros((0,), dtype=torch.long, device=device)
                    result["flat_roi_scores"] = pts_feats.new_zeros((0,))
                    result["flat_roi_labels"] = torch.zeros((0,), dtype=torch.long, device=device)

            except Exception:
                device = pts_feats.device
                result["flat_rois"] = pts_feats.new_zeros((0, 7))
                result["flat_roi_batch_indices"] = torch.zeros((0,), dtype=torch.long, device=device)
                result["flat_roi_scores"] = pts_feats.new_zeros((0,))
                result["flat_roi_labels"] = torch.zeros((0,), dtype=torch.long, device=device)

        return result
