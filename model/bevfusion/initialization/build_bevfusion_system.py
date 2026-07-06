"""Build the trainable FusionRepo BEVFusion system around real OpenMMLab models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common  # noqa: F401
import fusion_unit  # noqa: F401
import utils  # noqa: F401
from model.bevfusion.bevfusion_framework import BEVFusionFramework
from model.bevfusion.module import (
    BEVFusionModule,
    CameraFeatureModule,
    CameraViewTransformModule,
    DetectionTaskModule,
    LiDARBEVModule,
)


MMDET_CONFIG_ROOT = Path(
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet/.mim/configs"
)
MMDET3D_CONFIG_ROOT = Path(
    "/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet3d/.mim/configs"
)
DEFAULT_IMAGE_CONFIG = str(
    MMDET_CONFIG_ROOT / "faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"
)
DEFAULT_IMAGE_CHECKPOINT = "checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth"
DEFAULT_IMAGE_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
    "faster_rcnn_r50_caffe_fpn_mstrain_1x_coco/"
    "faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth"
)
DEFAULT_POINT_CLOUD_CONFIG = str(
    MMDET3D_CONFIG_ROOT / "pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
)
DEFAULT_POINT_CLOUD_CHECKPOINT = (
    "checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
)
DEFAULT_POINT_CLOUD_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/pv_rcnn/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class/"
    "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth"
)
DEFAULT_CLASS_NAMES = ("Pedestrian", "Cyclist", "Car")
DEFAULT_POINT_CLOUD_RANGE = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)


def ensure_checkpoint(path: str, url: str, minimum_bytes: int = 10 * 1024 * 1024) -> str:
    """Return a complete checkpoint, downloading it when absent or truncated."""
    checkpoint = Path(path)
    if checkpoint.exists() and checkpoint.stat().st_size >= int(minimum_bytes):
        return str(checkpoint)
    if checkpoint.exists():
        checkpoint.unlink()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["wget", "-O", str(checkpoint), url], check=True)
    except Exception as exc:
        if checkpoint.exists():
            checkpoint.unlink()
        raise RuntimeError(
            f"checkpoint download failed; URL={url}; path={checkpoint}"
        ) from exc
    if checkpoint.stat().st_size < int(minimum_bytes):
        checkpoint.unlink()
        raise RuntimeError(f"downloaded checkpoint is truncated; URL={url}; path={checkpoint}")
    return str(checkpoint)


class MMDetImageFeatureAdapter(nn.Module):
    """Expose real Faster R-CNN FPN features for ``(B,3,H,W)`` RGB images."""

    def __init__(self, config: str, checkpoint: str, device: str):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet")
        self.model = init_detector(config, checkpoint, device=device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        from mmdet.structures import DetDataSample

        height, width = images.shape[-2:]
        samples = []
        for _ in range(images.shape[0]):
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "ori_shape": (height, width),
                    "img_shape": (height, width),
                    "pad_shape": (height, width),
                    "scale_factor": (1.0, 1.0),
                }
            )
            samples.append(sample)
        processed = self.model.data_preprocessor(
            {"inputs": images, "data_samples": samples}, training=False
        )
        return self.model.extract_feat(processed["inputs"])


class MMDet3DDetectorAdapter(nn.Module):
    """Run real PV-RCNN inference and expose LiDAR boxes/scores/class labels."""

    def __init__(self, config: str, checkpoint: str, device: str, score_threshold: float = 0.05):
        super().__init__()
        from mmengine.registry import init_default_scope
        from mmdet3d.apis import init_model
        from mmdet3d.utils import register_all_modules

        register_all_modules()
        init_default_scope("mmdet3d")
        self.model = init_model(config, checkpoint, device=device)
        self.model.eval()
        self.device_name = device
        self.score_threshold = float(score_threshold)
        metadata = getattr(self.model, "dataset_meta", {}) or {}
        self.class_names = tuple(metadata.get("classes", DEFAULT_CLASS_NAMES))

    @torch.no_grad()
    def forward(self, batch: dict):
        from mmengine.registry import init_default_scope
        from mmdet3d.apis import inference_detector

        init_default_scope("mmdet3d")
        outputs = []
        for path in batch["point_paths"]:
            result = inference_detector(self.model, str(path))
            sample = result[0] if isinstance(result, tuple) else result
            if isinstance(sample, (list, tuple)):
                sample = sample[0]
            instances = sample.pred_instances_3d
            boxes = instances.bboxes_3d.tensor[:, :7].to(self.device_name)
            scores = instances.scores_3d.to(self.device_name)
            labels = instances.labels_3d.to(self.device_name)
            keep = scores >= self.score_threshold
            outputs.append(
                {
                    "boxes_3d": boxes[keep],
                    "scores_3d": scores[keep],
                    "labels_3d": labels[keep],
                }
            )
        return outputs


def _freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _trainable_parameters(module: nn.Module):
    return (parameter for parameter in module.parameters() if parameter.requires_grad)


def _keep_frozen_eval(model: BEVFusionFramework) -> None:
    model.camera_feature_module.image_adapter.eval()
    model.detection_task_module.detector_adapter.eval()


class BEVFusionCriterion(nn.Module):
    """Supervise fusion-conditioned box/score refinement with KITTI LiDAR GT boxes."""

    def __init__(self, box_weight: float = 1.0, score_weight: float = 0.25):
        super().__init__()
        self.box_weight = float(box_weight)
        self.score_weight = float(score_weight)

    def forward(self, output: dict, batch: dict) -> dict:
        box_losses = []
        score_losses = []
        for prediction, gt_boxes, gt_labels in zip(
            output["predictions"], batch["gt_boxes_3d"], batch["gt_labels_3d"]
        ):
            boxes = prediction["boxes_3d"]
            scores = prediction["scores_3d"]
            labels = prediction["labels_3d"]
            gt_boxes = gt_boxes.to(boxes)
            gt_labels = gt_labels.to(labels)
            matched_pred = []
            matched_gt = []
            for gt_index in range(gt_boxes.shape[0]):
                candidates = torch.where(labels == gt_labels[gt_index])[0]
                if candidates.numel() == 0:
                    continue
                distances = torch.linalg.vector_norm(
                    boxes[candidates, :2] - gt_boxes[gt_index, :2], dim=1
                )
                matched_pred.append(candidates[distances.argmin()])
                matched_gt.append(gt_index)
            if matched_pred:
                pred_index = torch.stack(matched_pred)
                gt_index = torch.tensor(matched_gt, device=boxes.device, dtype=torch.long)
                box_losses.append(F.smooth_l1_loss(boxes[pred_index], gt_boxes[gt_index]))
                score_losses.append(
                    F.binary_cross_entropy(scores[pred_index].clamp(1e-5, 1 - 1e-5), torch.ones_like(scores[pred_index]))
                )
        regularizer = output["box_deltas"].square().mean() + output["score_deltas"].square().mean()
        loss_box = torch.stack(box_losses).mean() if box_losses else regularizer * 0.0
        loss_score = torch.stack(score_losses).mean() if score_losses else regularizer * 0.0
        loss = self.box_weight * loss_box + self.score_weight * loss_score + 1e-4 * regularizer
        return {"loss": loss, "loss_box": loss_box.detach(), "loss_score": loss_score.detach()}


class BEVFusionPostProcessor:
    """Convert LiDAR predictions to official KITTI camera-coordinate annotations."""

    def __init__(self, class_names=DEFAULT_CLASS_NAMES):
        self.class_names = tuple(class_names)

    @staticmethod
    def _to44(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape == (4, 4):
            return matrix
        output = torch.eye(4, device=matrix.device, dtype=matrix.dtype)
        output[: matrix.shape[0], : matrix.shape[1]] = matrix
        return output

    @staticmethod
    def _corners_lidar(boxes: torch.Tensor) -> torch.Tensor:
        template = boxes.new_tensor(
            [[.5,.5,0],[.5,-.5,0],[-.5,-.5,0],[-.5,.5,0],
             [.5,.5,1],[.5,-.5,1],[-.5,-.5,1],[-.5,.5,1]]
        )
        corners = template.unsqueeze(0) * boxes[:, None, 3:6]
        cosine, sine = torch.cos(boxes[:, 6]), torch.sin(boxes[:, 6])
        rotation = boxes.new_zeros((boxes.shape[0], 3, 3))
        rotation[:, 0, 0], rotation[:, 0, 1] = cosine, -sine
        rotation[:, 1, 0], rotation[:, 1, 1] = sine, cosine
        rotation[:, 2, 2] = 1
        return corners @ rotation.transpose(1, 2) + boxes[:, None, :3]

    def _convert_sample(self, prediction: dict, calib: dict, image_shape):
        boxes = prediction["boxes_3d"]
        device, dtype = boxes.device, boxes.dtype
        p2 = calib["P2"].to(device=device, dtype=dtype)[:3, :4]
        rect = self._to44(calib["R0_rect"].to(device=device, dtype=dtype))
        transform = self._to44(calib["Tr_velo2cam"].to(device=device, dtype=dtype))
        lidar_to_camera = rect @ transform
        corners = self._corners_lidar(boxes)
        ones = torch.ones((*corners.shape[:2], 1), device=device, dtype=dtype)
        corners_h = torch.cat([corners, ones], dim=-1)
        corners_camera = corners_h @ lidar_to_camera.t()
        pixels_h = corners_camera @ p2.t()
        depth = pixels_h[..., 2].clamp_min(1e-5)
        pixels = pixels_h[..., :2] / depth.unsqueeze(-1)
        bbox_min = pixels.min(dim=1).values
        bbox_max = pixels.max(dim=1).values
        height, width = image_shape
        bbox = torch.cat([bbox_min, bbox_max], dim=1)
        bbox[:, 0::2].clamp_(0, max(int(width) - 1, 0))
        bbox[:, 1::2].clamp_(0, max(int(height) - 1, 0))

        centers_h = torch.cat([boxes[:, :3], torch.ones((boxes.shape[0], 1), device=device, dtype=dtype)], dim=1)
        locations = (centers_h @ lidar_to_camera.t())[:, :3]
        valid = (
            (bbox_min[:, 0] < int(width))
            & (bbox_min[:, 1] < int(height))
            & (bbox_max[:, 0] > 0)
            & (bbox_max[:, 1] > 0)
            & (locations[:, 2] > 0)
        )
        boxes = boxes[valid]
        bbox = bbox[valid]
        locations = locations[valid]
        dimensions = torch.stack([boxes[:, 5], boxes[:, 4], boxes[:, 3]], dim=1)
        rotation_y = -boxes[:, 6] - math.pi / 2.0
        rotation_y = (rotation_y + math.pi) % (2 * math.pi) - math.pi
        alpha = rotation_y - torch.atan2(locations[:, 0], locations[:, 2])
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        labels = prediction["labels_3d"][valid].detach().cpu().numpy()
        names = np.asarray(
            [self.class_names[int(label)] if int(label) < len(self.class_names) else self.class_names[0] for label in labels],
            dtype=object,
        )
        count = boxes.shape[0]
        return {
            "name": names,
            "truncated": np.zeros(count, dtype=np.float32),
            "occluded": np.zeros(count, dtype=np.int64),
            "alpha": alpha.detach().cpu().numpy(),
            "bbox": bbox.detach().cpu().numpy(),
            "dimensions": dimensions.detach().cpu().numpy(),
            "location": locations.detach().cpu().numpy(),
            "rotation_y": rotation_y.detach().cpu().numpy(),
            "score": prediction["scores_3d"][valid].detach().cpu().numpy(),
        }

    def __call__(self, predictions, batch: dict):
        return [
            self._convert_sample(prediction, calib, shape)
            for prediction, calib, shape in zip(
                predictions, batch["calibs"], batch["image_shapes"]
            )
        ]


@dataclass
class BEVFusionSystem:
    model: BEVFusionFramework
    criterion: BEVFusionCriterion
    postprocessor: BEVFusionPostProcessor
    optimizer: torch.optim.Optimizer
    device: str
    image_config: str
    image_checkpoint: str
    point_cloud_config: str
    point_cloud_checkpoint: str
    class_names: tuple[str, ...]
    grad_clip_norm: Optional[float] = 10.0

    def trainable_counts(self) -> dict:
        image = self.model.camera_feature_module.image_adapter
        point = self.model.detection_task_module.detector_adapter
        return {
            "image_trainable": sum(p.numel() for p in image.parameters() if p.requires_grad),
            "point_cloud_trainable": sum(p.numel() for p in point.parameters() if p.requires_grad),
            "optimizer_parameters": sum(p.numel() for group in self.optimizer.param_groups for p in group["params"]),
        }


def build_bevfusion_system(
    device: str = "cuda:0",
    image_config: str = DEFAULT_IMAGE_CONFIG,
    image_checkpoint: str = DEFAULT_IMAGE_CHECKPOINT,
    point_cloud_config: str = DEFAULT_POINT_CLOUD_CONFIG,
    point_cloud_checkpoint: str = DEFAULT_POINT_CLOUD_CHECKPOINT,
    fusion_checkpoint: str | None = None,
    lr: float = 1e-4,
) -> BEVFusionSystem:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if not Path(image_config).is_file():
        raise FileNotFoundError(f"mmdet config not found: {image_config}")
    if not Path(point_cloud_config).is_file():
        raise FileNotFoundError(f"mmdet3d config not found: {point_cloud_config}")
    image_checkpoint = ensure_checkpoint(image_checkpoint, DEFAULT_IMAGE_CHECKPOINT_URL)
    point_cloud_checkpoint = ensure_checkpoint(
        point_cloud_checkpoint, DEFAULT_POINT_CLOUD_CHECKPOINT_URL
    )
    image_adapter = MMDetImageFeatureAdapter(image_config, image_checkpoint, device)
    detector_adapter = MMDet3DDetectorAdapter(
        point_cloud_config, point_cloud_checkpoint, device
    )
    class_names = tuple(detector_adapter.class_names)
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(
            f"unexpected KITTI class order {class_names}; expected {DEFAULT_CLASS_NAMES} from checkpoint config"
        )
    channels = 32
    model = BEVFusionFramework(
        camera_feature_module=CameraFeatureModule(image_adapter, [256] * 5, channels),
        camera_view_transform_module=CameraViewTransformModule(
            channels, 8, 2.0, 50.0, 80, 80, DEFAULT_POINT_CLOUD_RANGE
        ),
        lidar_bev_module=LiDARBEVModule(
            DEFAULT_POINT_CLOUD_RANGE,
            (0.88, 1.0, 2.0),
            (2, 80, 80),
            channels,
        ),
        bev_fusion_module=BEVFusionModule(channels, channels, channels),
        detection_task_module=DetectionTaskModule(
            detector_adapter, channels, len(class_names)
        ),
    ).to(device)
    _freeze_module(image_adapter)
    _freeze_module(detector_adapter)
    if fusion_checkpoint:
        state = torch.load(fusion_checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
    criterion = BEVFusionCriterion().to(device)
    parameters = list(_trainable_parameters(model))
    if not parameters:
        raise RuntimeError("BEVFusion has no trainable newly added parameters")
    optimizer = torch.optim.AdamW(parameters, lr=float(lr), weight_decay=1e-4)
    system = BEVFusionSystem(
        model=model,
        criterion=criterion,
        postprocessor=BEVFusionPostProcessor(class_names),
        optimizer=optimizer,
        device=device,
        image_config=image_config,
        image_checkpoint=image_checkpoint,
        point_cloud_config=point_cloud_config,
        point_cloud_checkpoint=point_cloud_checkpoint,
        class_names=class_names,
    )
    counts = system.trainable_counts()
    if counts["image_trainable"] or counts["point_cloud_trainable"] or counts["optimizer_parameters"] <= 0:
        raise RuntimeError(f"invalid freezing/optimizer partition: {counts}")
    return system


def _move_batch(batch: dict, device: str) -> dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            output[key] = [item.to(device) for item in value]
        elif key == "calibs":
            output[key] = [
                {name: matrix.to(device) for name, matrix in calib.items()} for calib in value
            ]
        else:
            output[key] = value
    return output


def train_step(system: BEVFusionSystem, batch: dict) -> dict:
    batch = _move_batch(batch, system.device)
    system.model.train()
    _keep_frozen_eval(system.model)
    system.optimizer.zero_grad(set_to_none=True)
    output = system.model.train_forward(batch)
    losses = system.criterion(output, batch)
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError(f"non-finite BEVFusion loss: {losses['loss'].item()}")
    losses["loss"].backward()
    if system.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            list(_trainable_parameters(system.model)), system.grad_clip_norm
        )
    system.optimizer.step()
    return {key: value.detach() for key, value in losses.items()}


@torch.no_grad()
def test_step(system: BEVFusionSystem, batch: dict) -> dict:
    batch = _move_batch(batch, system.device)
    system.model.eval()
    output = system.model.test_forward(batch)
    annotations = system.postprocessor(output["predictions"], batch)
    return {"predictions": output["predictions"], "annotations": annotations}
