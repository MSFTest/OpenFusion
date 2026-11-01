# -*- coding: utf-8 -*-
"""
PointCloudFeatureInterface

纯点云特征提取接口，正确处理 mmdet3d VoxelNet/PointPillars 的数据流。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any, Optional, List, Tuple


def _clone_tensor_tree(value):
    """Copy input containers and tensors before an independent model forward."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    return value


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
        point_cloud_range: Optional[Tuple[float, float, float, float, float, float]] = None,
        target_class_name: Optional[str] = None,
        score_input_type: str = "probability",
        point_cloud_feature_level: str = "voxel",
        proposal_source: str = "final",
    ):
        super().__init__()
        self.extractor = extractor
        self.feat_index = feat_index
        self.use_detection = use_detection
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        if point_cloud_feature_level not in {"voxel", "middle_stage3", "middle_stage4"}:
            raise ValueError(
                "point_cloud_feature_level must be 'voxel', 'middle_stage3', or 'middle_stage4'"
            )
        self.point_cloud_feature_level = point_cloud_feature_level
        if proposal_source not in {"final", "rpn"}:
            raise ValueError("proposal_source must be 'final' or 'rpn'")
        self.proposal_source = proposal_source
        if score_input_type not in ("probability", "logit"):
            raise ValueError("score_input_type must be 'probability' or 'logit'")
        self.score_input_type = score_input_type
        self.target_class_index = None
        if target_class_name is not None:
            dataset_meta = getattr(extractor, "dataset_meta", None)
            classes = dataset_meta.get("classes") if isinstance(dataset_meta, dict) else None
            if not classes:
                raise ValueError("target_class_name requires extractor.dataset_meta['classes']")
            if target_class_name not in classes:
                raise ValueError(f"target class {target_class_name!r} is absent from dataset_meta['classes']={classes!r}")
            self.target_class_index = classes.index(target_class_name)
        self.target_class_name = target_class_name

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
        if hasattr(feat, "features") and hasattr(feat, "indices"):
            return "sparse_tensor"
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
        return f"tensor_{feat.dim()}d"

    def _predict_middle_stage(self, inputs, data_samples, batch_size):
        stage_name, stage_index, stride = {
            "middle_stage3": ("middle_stage3", 2, 4),
            "middle_stage4": ("middle_stage4", 3, 8),
        }[self.point_cloud_feature_level]
        middle_encoder = getattr(self.extractor, "middle_encoder", None)
        encoder_layers = getattr(middle_encoder, "encoder_layers", None)
        try:
            stage = encoder_layers[stage_index]
        except (TypeError, IndexError, KeyError):
            raise ValueError(
                f"point detector does not expose middle_encoder.encoder_layers for {stage_name}"
            ) from None
        if not isinstance(stage, nn.Module):
            raise ValueError(f"point detector {stage_name} is not a module")

        captured = []

        def capture(_module, _inputs, output):
            if captured:
                raise RuntimeError(f"PartA2 {stage_name} ran more than once during predict")
            features = getattr(output, "features", None)
            indices = getattr(output, "indices", None)
            spatial_shape = getattr(output, "spatial_shape", None)
            if not torch.is_tensor(features) or features.ndim != 2:
                raise ValueError(f"PartA2 {stage_name} must return SparseConvTensor features shaped (N, C)")
            if features.shape[1] != 64:
                raise ValueError(
                    f"PartA2 {stage_name} must return 64 channels, got {features.shape[1]}"
                )
            if not torch.is_tensor(indices) or indices.ndim != 2 or indices.shape != (features.shape[0], 4):
                raise ValueError(f"PartA2 {stage_name} indices must have shape (N, 4) aligned with features")
            try:
                spatial_shape = tuple(int(value) for value in spatial_shape)
            except (TypeError, ValueError):
                raise ValueError(f"PartA2 {stage_name} must expose a three-dimensional spatial_shape") from None
            if len(spatial_shape) != 3 or any(value <= 0 for value in spatial_shape):
                raise ValueError(f"PartA2 {stage_name} must expose a positive three-dimensional spatial_shape")
            if not torch.isfinite(features).all():
                raise ValueError(f"PartA2 {stage_name} features must be finite")
            if indices.is_floating_point() and (
                not torch.isfinite(indices).all() or not torch.equal(indices, indices.round())
            ):
                raise ValueError(f"PartA2 {stage_name} indices must be finite integers")
            indices = indices.to(device=features.device, dtype=torch.long)
            if indices.numel():
                if ((indices[:, 0] < 0) | (indices[:, 0] >= batch_size)).any():
                    raise ValueError(f"PartA2 {stage_name} batch indices are outside the input batch")
                limits = indices.new_tensor(spatial_shape)
                if ((indices[:, 1:] < 0) | (indices[:, 1:] >= limits)).any():
                    raise ValueError(f"PartA2 {stage_name} coordinates are outside spatial_shape")
            captured.append((features, indices, spatial_shape))

        handle = stage.register_forward_hook(capture)
        try:
            if self.use_detection and self.proposal_source == "rpn":
                rpn_head = getattr(self.extractor, "rpn_head", None)
                if rpn_head is None or not hasattr(rpn_head, "predict"):
                    raise ValueError("point detector does not expose rpn_head.predict")
                feats_dict = self.extractor.extract_feat(inputs)
                predictions = rpn_head.predict(feats_dict, data_samples)
            elif self.use_detection:
                predictions = self.extractor.predict(inputs, data_samples)
            else:
                predictions = None
                self.extractor.extract_feat(inputs)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise ValueError(f"predict did not capture the requested PartA2 feature {stage_name}")
        features, indices, spatial_shape = captured[0]
        voxel_size, point_cloud_range = self._infer_voxel_params()
        if voxel_size is None or point_cloud_range is None:
            raise ValueError(f"voxel_size and point_cloud_range are required for {stage_name} coordinates")
        voxel_size_t = features.new_tensor(voxel_size, dtype=torch.float32) * stride
        pc_min_t = features.new_tensor(point_cloud_range[:3], dtype=torch.float32)
        xyz_indices = indices[:, [3, 2, 1]].to(dtype=torch.float32)
        centers = (xyz_indices + 0.5) * voxel_size_t + pc_min_t
        if not torch.isfinite(centers).all():
            raise ValueError(f"PartA2 {stage_name} centers must be finite")
        return predictions, features, indices, centers

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
            dict with pts_feats, flat_point_batch_indices, voxel_coords, voxels, voxel_num_points, flat_rois, flat_roi_batch_indices
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
        middle_predictions = None
        proposal_predictions = None
        if self.point_cloud_feature_level != "voxel":
            middle_predictions, pts_feats, voxel_coords, voxel_centers_lidar = self._predict_middle_stage(
                _clone_tensor_tree(batch_inputs_dict), batch_data_samples, batch_size
            )
            feats = {self.point_cloud_feature_level: pts_feats}
            selected_feat = pts_feats
        elif (
            voxels is not None
            and voxel_coords is not None
            and voxel_num_points is not None
            and hasattr(self.extractor, "voxel_encoder")
        ):
            try:
                # Some legacy PillarFeatureNet implementations decorate point
                # coordinates in-place. Keep this auxiliary aligned-feature
                # pass from changing the inputs used by detector inference.
                voxel_input = _clone_tensor_tree(batch_inputs_dict["voxels"])
                aligned_voxel_features = self.extractor.voxel_encoder(
                    voxel_input["voxels"],
                    voxel_input["num_points"],
                    voxel_input["coors"]
                )
            except Exception as exc:
                raise RuntimeError("point detector voxel feature extraction failed") from exc

        # extract_feat and predict are independent forwards, as in an official
        # test_step's single predict call. Give each a fresh copy because voxel
        # encoders may modify preprocessed tensors in-place.
        if self.point_cloud_feature_level == "voxel":
            feats = self.extractor.extract_feat(_clone_tensor_tree(batch_inputs_dict))
            if self.use_detection and self.proposal_source == "rpn":
                rpn_head = getattr(self.extractor, "rpn_head", None)
                if rpn_head is None or not hasattr(rpn_head, "predict"):
                    raise ValueError("point detector does not expose rpn_head.predict")
                proposal_predictions = rpn_head.predict(feats, batch_data_samples)

        if self.point_cloud_feature_level != "voxel":
            pass
        elif aligned_voxel_features is not None:
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

        if self.point_cloud_feature_level == "voxel":
            selected_feat = pts_feats
        feature_layout = self._feature_layout(selected_feat)

        # 如果是4D特征图 (B, H, W, C)，展平为2D (B*H*W, C)
        flat_point_batch_indices = None
        if self.point_cloud_feature_level != "voxel":
            flat_point_batch_indices = voxel_coords[:, 0].to(device=pts_feats.device, dtype=torch.long)
        if pts_feats.dim() == 4:
            batch_count, height, width, _ = pts_feats.shape
            flat_point_batch_indices = torch.arange(batch_count, device=pts_feats.device).repeat_interleave(height * width)
        if pts_feats.dim() == 4:
            B, H, W, C = pts_feats.shape
            pts_feats = pts_feats.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            pts_feats = pts_feats.view(B, C, H * W)  # (B, C, H*W)
            pts_feats = pts_feats.permute(0, 2, 1).contiguous()  # (B, H*W, C)
            pts_feats = pts_feats.view(-1, C)  # (B*H*W, C)

        if pts_feats.dim() != 2:
            raise ValueError("point features must flatten to (N, C) before assigning batch indices")
        if flat_point_batch_indices is None:
            if voxel_coords is not None:
                if not torch.is_tensor(voxel_coords) or voxel_coords.ndim != 2 or voxel_coords.shape[1] < 1:
                    raise ValueError("voxel coordinates must include a batch-index column")
                flat_point_batch_indices = voxel_coords[:, 0].to(device=pts_feats.device, dtype=torch.long)
            elif batch_size == 1:
                flat_point_batch_indices = torch.zeros((pts_feats.shape[0],), device=pts_feats.device, dtype=torch.long)
            elif pts_feats.shape[0] == 0:
                flat_point_batch_indices = torch.zeros((0,), device=pts_feats.device, dtype=torch.long)
            else:
                raise ValueError("flat point batch indices are required for multiple samples")
        if flat_point_batch_indices.numel() != pts_feats.shape[0]:
            raise ValueError("flat point batch indices must align with point features")
        flat_point_batch_indices = flat_point_batch_indices.to(device=pts_feats.device, dtype=torch.long)
        if ((flat_point_batch_indices < 0) | (flat_point_batch_indices >= batch_size)).any():
            raise ValueError("flat point batch indices must be within the input batch")

        if self.point_cloud_feature_level == "voxel":
            voxel_centers_lidar = self._compute_voxel_centers_lidar(voxel_coords)
        result = {
            "pts_feats": pts_feats,
            "flat_point_batch_indices": flat_point_batch_indices,
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
                # Data samples have a single consumer (predict); only the
                # tensor inputs need fresh copies for independent forwards.
                preds = middle_predictions
                if preds is None:
                    preds = proposal_predictions
                if preds is None:
                    preds = self.extractor.predict(
                        _clone_tensor_tree(batch_inputs_dict),
                        batch_data_samples
                    )
            except Exception as exc:
                batch_indices = list(range(batch_size))
                raise RuntimeError(
                    f"point detector predict failed for batch indices={batch_indices}: {exc}"
                ) from exc
            if not isinstance(preds, (list, tuple)) or len(preds) != batch_size:
                result_count = len(preds) if isinstance(preds, (list, tuple)) else type(preds).__name__
                raise ValueError(
                    "point detector predict must return one result per sample: "
                    f"received {result_count} results for batch_size={batch_size}"
                )

            # 解析检测框；成功返回空预测是有效结果，不应和 predict 异常混淆。
            flat_rois_list = []
            flat_batch_idx_list = []
            flat_score_list = []
            flat_label_list = []

            for b, pred in enumerate(preds):
                inst = getattr(pred, "pred_instances_3d", pred)
                if hasattr(inst, "bboxes_3d"):
                    boxes = inst.bboxes_3d
                    if hasattr(boxes, "tensor"):
                        boxes = boxes.tensor
                    if boxes.numel() > 0:
                        scores = getattr(inst, "scores_3d", None)
                        if scores is None:
                            scores = getattr(inst, "scores", None)
                        labels = getattr(inst, "labels_3d", None)
                        if labels is None:
                            labels = getattr(inst, "labels", None)
                        if self.target_class_name is not None and (scores is None or labels is None):
                            raise ValueError("class filtering requires aligned detector scores and labels")
                        if scores is not None:
                            scores = scores.to(device=boxes.device).reshape(-1)
                            if scores.numel() != boxes.shape[0]:
                                raise ValueError("detector scores must align with boxes")
                            if not torch.isfinite(scores).all():
                                raise ValueError("detector scores must be finite")
                            if self.score_input_type == "logit":
                                scores = scores.sigmoid()
                            elif ((scores < 0) | (scores > 1)).any():
                                raise ValueError("detector scores declared as probabilities must be within [0, 1]")
                        if labels is not None:
                            labels = labels.to(device=boxes.device, dtype=torch.long).reshape(-1)
                            if labels.numel() != boxes.shape[0]:
                                raise ValueError("detector labels must align with boxes")
                        if self.target_class_name is not None:
                            keep = labels == self.target_class_index
                            boxes = boxes[keep]
                            scores = scores[keep]
                            labels = torch.zeros((int(keep.sum().item()),), dtype=torch.long, device=boxes.device)
                        if boxes.numel() == 0:
                            continue
                        # MMDet3D LiDAR boxes default to bottom-center origin;
                        # MPCF ROIs, GT boxes, and box deltas use geometric centers.
                        boxes = boxes.clone()
                        boxes[:, 2] += boxes[:, 5] / 2
                        flat_rois_list.append(boxes)
                        flat_batch_idx_list.append(
                            torch.full((boxes.shape[0],), b,
                                      dtype=torch.long, device=boxes.device)
                        )
                        if scores is not None:
                            flat_score_list.append(scores)
                        if labels is not None:
                            flat_label_list.append(labels)

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

        return result
