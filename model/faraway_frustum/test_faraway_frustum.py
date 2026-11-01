# -*- coding: utf-8 -*-
"""Faraway-Frustum KITTI train/test 入口。"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.faraway_frustum.initialization.build_faraway_frustum_system import (
    LOSS_VERSION,
    build_faraway_frustum_system,
    raw_test_step,
    test_step,
    train_step,
    verify_frozen_modules,
)


CLASS_TO_ID = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}


def normalize_path(path: str) -> str:
    path = path.strip().strip('"').strip("'")
    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        rest = path[2:].replace("\\", "/")
        return f"/mnt/{path[0].lower()}{rest}"
    return path


def read_split(split_file: str, max_samples: int = 0) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()]
    if max_samples and max_samples > 0:
        sample_ids = sample_ids[: int(max_samples)]
    return sample_ids


def read_calib(path: str) -> Dict[str, torch.Tensor]:
    raw: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            raw[key] = np.array([float(x) for x in values.split()], dtype=np.float32)
    return {
        "P2": torch.from_numpy(raw["P2"].reshape(3, 4)).float(),
        "R0_rect": torch.from_numpy(raw["R0_rect"].reshape(3, 3)).float(),
        "Tr_velo_to_cam": torch.from_numpy(raw["Tr_velo_to_cam"].reshape(3, 4)).float(),
    }


def read_label(path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    boxes_2d, boxes_3d, labels = [], [], []
    names, truncated, occluded, alpha, bbox, dimensions, location, rotation_y = [], [], [], [], [], [], [], []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 15 or parts[0] not in CLASS_TO_ID:
                    continue
                cls = parts[0]
                vals = [float(x) for x in parts[1:15]]
                x1, y1, x2, y2 = vals[3:7]
                h, w, l = vals[7:10]
                x, y, z = vals[10:13]
                ry = vals[13]
                boxes_2d.append([x1, y1, x2, y2])
                boxes_3d.append([x, y, z, w, l, h, ry])
                labels.append(CLASS_TO_ID[cls])
                names.append(cls)
                truncated.append(vals[0])
                occluded.append(int(vals[1]))
                alpha.append(vals[2])
                bbox.append([x1, y1, x2, y2])
                dimensions.append([h, w, l])
                location.append([x, y, z])
                rotation_y.append(ry)
    anno = {
        "name": np.array(names),
        "truncated": np.array(truncated, dtype=np.float32),
        "occluded": np.array(occluded, dtype=np.int64),
        "alpha": np.array(alpha, dtype=np.float32),
        "bbox": np.array(bbox, dtype=np.float32).reshape(-1, 4),
        "dimensions": np.array(dimensions, dtype=np.float32).reshape(-1, 3),
        "location": np.array(location, dtype=np.float32).reshape(-1, 3),
        "rotation_y": np.array(rotation_y, dtype=np.float32),
    }
    return (
        torch.tensor(boxes_2d, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(boxes_3d, dtype=torch.float32).reshape(-1, 7),
        torch.tensor(labels, dtype=torch.long),
        anno,
    )


def sample_has_faraway_gt(
    kitti_root: str,
    sample_id: str,
    class_ids: List[int],
    min_depth: float,
) -> bool:
    """判断样本是否包含指定类别的远距离 GT。

    Args:
        kitti_root: KITTI 根目录。
        sample_id: 样本 ID。
        class_ids: 允许的类别 ID；空列表表示所有 KITTI 三类。
        min_depth: camera z 最小深度，单位米。

    Returns:
        bool，若存在 ``z >= min_depth`` 的目标则为 True。
    """
    label_path = os.path.join(kitti_root, "training", "label_2", f"{sample_id}.txt")
    _, boxes_3d, labels, _ = read_label(label_path)
    if boxes_3d.numel() == 0:
        return False
    if class_ids:
        allowed = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in class_ids:
            allowed = allowed | (labels == int(class_id))
    else:
        allowed = torch.ones_like(labels, dtype=torch.bool)
    return bool((allowed & (boxes_3d[:, 2] >= float(min_depth))).any().item())


def prepare_train_sample_ids(
    kitti_root: str,
    split_file: str,
    max_samples: int,
    train_class_ids: List[int],
    faraway_gt_min_depth: float,
    repeat_faraway_samples: int,
) -> List[str]:
    """读取训练 split，并可按远距离 GT 过滤和过采样。

    Returns:
        sample_ids: 训练样本 ID 列表。若启用过滤，列表只包含含远距离 GT 的样本，
        并按 ``repeat_faraway_samples`` 重复。
    """
    sample_ids = read_split(split_file, 0)
    if faraway_gt_min_depth > 0.0:
        sample_ids = [
            sample_id for sample_id in sample_ids
            if sample_has_faraway_gt(kitti_root, sample_id, train_class_ids, faraway_gt_min_depth)
        ]
        if not sample_ids:
            raise RuntimeError(
                f"No training samples contain requested faraway GT: "
                f"min_depth={faraway_gt_min_depth} class_ids={train_class_ids}"
            )
        sample_ids = sample_ids * max(int(repeat_faraway_samples), 1)
    if max_samples and max_samples > 0:
        sample_ids = sample_ids[: int(max_samples)]
    return sample_ids


def epoch_sample_order(sample_ids: List[str], seed: int, epoch: int) -> List[str]:
    """返回不修改原列表的可复现 epoch 训练顺序。"""
    ordered = list(sample_ids)
    random.Random(int(seed) + int(epoch) * 1_000_003).shuffle(ordered)
    return ordered


def make_batch(kitti_root: str, sample_id: str, device: str, split: str = "training") -> Dict[str, Any]:
    base = os.path.join(kitti_root, split)
    image_path = os.path.join(base, "image_2", f"{sample_id}.png")
    points_path = os.path.join(base, "velodyne", f"{sample_id}.bin")
    calib_path = os.path.join(base, "calib", f"{sample_id}.txt")
    label_path = os.path.join(base, "label_2", f"{sample_id}.txt")
    if not os.path.exists(image_path) and split == "training":
        base = os.path.join(kitti_root, "testing")
        image_path = os.path.join(base, "image_2", f"{sample_id}.png")
        points_path = os.path.join(base, "velodyne", f"{sample_id}.bin")
        calib_path = os.path.join(base, "calib", f"{sample_id}.txt")
        label_path = os.path.join(base, "label_2", f"{sample_id}.txt")
    image = Image.open(image_path)
    image_shape = (int(image.height), int(image.width))
    points = torch.from_numpy(np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)).to(device)
    calib = {key: value.to(device) for key, value in read_calib(calib_path).items()}
    gt_boxes_2d, gt_boxes_3d_camera, gt_labels, gt_anno = read_label(label_path)
    return {
        "sample_id": sample_id,
        "image_path": image_path,
        "points_path": points_path,
        "points": points,
        "calib": calib,
        "image_shape": image_shape,
        "gt_boxes_2d": gt_boxes_2d.to(device),
        "gt_boxes_3d_camera": gt_boxes_3d_camera.to(device),
        "gt_labels": gt_labels.to(device),
        "gt_anno": gt_anno,
    }


def atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(path: str, system: Any, epoch: int, best_loss: float) -> None:
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "loss_version": LOSS_VERSION,
            "best_loss": float(best_loss),
            "model_state_scope": "fusion_modules",
            "model": {
                "faraway_refinement": system.model.faraway_refinement.state_dict(),
                "near_image_score_fusion": system.model.near_image_score_fusion.state_dict(),
            },
            "optimizer": system.optimizer.state_dict(),
        },
        path,
    )


def format_loss_metrics(losses: Dict[str, torch.Tensor]) -> str:
    parts = []
    for key in (
        "loss_center",
        "loss_depth",
        "loss_y",
        "loss_size",
        "loss_yaw",
        "loss_score",
        "loss_near_score",
        "loss_near_rank",
        "loss_near_preserve",
        "num_faraway",
        "num_matched",
        "num_score_positive",
        "num_near_positive",
        "num_near_negative",
        "num_near_ignored",
        "mean_near_max_iou",
        "mean_match_distance",
    ):
        if key not in losses:
            continue
        value = losses[key]
        if not torch.is_tensor(value):
            parts.append(f"{key}={value}")
        elif value.dtype in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8):
            parts.append(f"{key}={int(value.item())}")
        else:
            parts.append(f"{key}={float(value.item()):.6f}")
    return " ".join(parts)


def load_checkpoint(
    path: str,
    system: Any,
    reset_optimizer: bool = False,
    reset_best_loss: bool = False,
) -> Tuple[int, float]:
    if not path:
        return 0, float("inf")
    ckpt = torch.load(path, map_location=system.device)
    scope = ckpt.get("model_state_scope", "legacy_faraway_refinement")
    skipped_shape = []
    missing = []
    unexpected = []
    calibration_missing = []

    def load_compatible(module: torch.nn.Module, state: Dict[str, torch.Tensor], prefix: str) -> None:
        current = module.state_dict()
        compatible = {}
        for key, value in state.items():
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                compatible[key] = value
            else:
                skipped_shape.append(f"{prefix}{key}")
        incompatible = module.load_state_dict(compatible, strict=False)
        missing.extend(f"{prefix}{key}" for key in incompatible.missing_keys)
        unexpected.extend(f"{prefix}{key}" for key in incompatible.unexpected_keys)

    if scope == "fusion_modules":
        model_state = ckpt["model"]
        load_compatible(
            system.model.faraway_refinement,
            model_state.get("faraway_refinement", {}),
            "faraway_refinement.",
        )
        load_compatible(
            system.model.near_image_score_fusion,
            model_state.get("near_image_score_fusion", {}),
            "near_image_score_fusion.",
        )
    else:
        load_compatible(system.model.faraway_refinement, ckpt["model"], "faraway_refinement.")
        calibration_missing = [
            f"near_image_score_fusion.{key}"
            for key in system.model.near_image_score_fusion.state_dict()
        ]
        missing.extend(calibration_missing)
    refine_missing = [key for key in missing if "ff_net" in key or "faraway" in key or "score_head" in key]
    if not calibration_missing:
        calibration_missing = [key for key in missing if "near_image_score_fusion" in key]
    print(
        f"loaded_checkpoint={path} epoch={int(ckpt.get('epoch', 0))} "
        f"loss_version={ckpt.get('loss_version', 'unknown')} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"refine_missing={len(refine_missing)} calibration_missing={len(calibration_missing)} "
        f"skipped_shape={len(skipped_shape)}"
    )
    if "optimizer" in ckpt and not reset_optimizer:
        system.optimizer.load_state_dict(ckpt["optimizer"])
    best_loss = float("inf") if reset_best_loss or ckpt.get("loss_version") != LOSS_VERSION else float(ckpt.get("best_loss", float("inf")))
    if reset_optimizer:
        print("optimizer_state=reset")
    if reset_best_loss or ckpt.get("loss_version") != LOSS_VERSION:
        print("best_loss=reset")
    return int(ckpt.get("epoch", 0)), best_loss


def count_kitti_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def prediction_delta(fused: Dict[str, torch.Tensor], raw: Dict[str, torch.Tensor]) -> Dict[str, float]:
    fused_boxes = fused["boxes_3d_camera"].detach().float().cpu()
    raw_boxes = raw["boxes_3d_camera"].detach().float().cpu()
    fused_scores = fused["scores"].detach().float().cpu()
    raw_scores = raw["scores"].detach().float().cpu()
    common = min(fused_boxes.shape[0], raw_boxes.shape[0])
    if common > 0:
        box_delta = (fused_boxes[:common] - raw_boxes[:common]).abs()
        score_delta = (fused_scores[:common] - raw_scores[:common]).abs()
        return {
            "mean_abs_box_delta": float(box_delta.mean().item()),
            "max_abs_box_delta": float(box_delta.max().item()),
            "mean_abs_score_delta": float(score_delta.mean().item()),
            "max_abs_score_delta": float(score_delta.max().item()),
            "raw_count": float(raw_boxes.shape[0]),
            "fused_count": float(fused_boxes.shape[0]),
            "faraway_extra_count": float(max(fused_boxes.shape[0] - raw_boxes.shape[0], 0)),
        }
    return {
        "mean_abs_box_delta": 0.0,
        "max_abs_box_delta": 0.0,
        "mean_abs_score_delta": 0.0,
        "max_abs_score_delta": 0.0,
        "raw_count": float(raw_boxes.shape[0]),
        "fused_count": float(fused_boxes.shape[0]),
        "faraway_extra_count": float(max(fused_boxes.shape[0] - raw_boxes.shape[0], 0)),
    }


def run_train(args: argparse.Namespace) -> None:
    if args.save_best_ckpt and (args.save_best_ckpt == args.save_last_ckpt or args.save_best_ckpt == args.save_ckpt):
        raise ValueError("save_best_ckpt must not be the same path as save_last_ckpt/save_ckpt")
    system = build_faraway_frustum_system(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        depth_thresholds=parse_depth_thresholds(args.depth_thresholds),
        image_score_thr=args.image_score_thr,
        max_2d_dets=args.max_2d_dets,
        bev_width_meters=args.bev_width_meters,
        bev_depth_meters=args.bev_depth_meters,
        bev_resolution=args.bev_resolution,
        faraway_score_mode=args.faraway_score_mode,
        faraway_image_geometry_alignment=args.faraway_image_geometry_alignment,
        train_class_ids=parse_train_classes(args.train_classes),
        faraway_infer_class_ids=parse_train_classes(args.faraway_infer_classes),
        faraway_score_thr=args.faraway_score_thr,
        faraway_near_suppress_radius=args.faraway_near_suppress_radius,
        criterion_match_radius=args.criterion_match_radius,
        yaw_loss_weight=args.yaw_loss_weight,
        score_loss_weight=args.score_loss_weight,
        train_with_detector_2d=args.train_with_detector_2d,
        match_2d_iou_thr=args.match_2d_iou_thr,
        score_positive_radius=args.score_positive_radius,
        score_raw_miss_radius=args.score_raw_miss_radius,
        paper_distance_fusion=args.paper_distance_fusion,
        pedestrian_far_distance=args.pedestrian_far_distance,
        car_far_distance=args.car_far_distance,
        faraway_nms_radius=args.faraway_nms_radius,
        faraway_topk_per_class=args.faraway_topk_per_class,
        car_score_thr=args.car_score_thr,
        point_cloud_model=args.point_cloud_model,
        point_cloud_config=args.point_cloud_config,
        point_cloud_checkpoint=args.point_cloud_ckpt,
        point_cloud_checkpoint_min_bytes=args.point_cloud_ckpt_min_bytes,
        near_image_score_fusion=args.near_image_score_fusion,
        near_image_score_power=args.near_image_score_power,
        near_image_score_min_factor=args.near_image_score_min_factor,
        near_image_score_trainable=args.near_image_score_trainable,
        near_image_score_residual_scale=args.near_image_score_residual_scale,
        near_score_loss_weight=args.near_score_loss_weight,
        near_score_preserve_weight=args.near_score_preserve_weight,
        near_rank_loss_weight=args.near_rank_loss_weight,
        near_rank_margin=args.near_rank_margin,
        near_negative_only_weight=args.near_negative_only_weight,
        near_positive_iou_thr=args.near_positive_iou_thr,
        near_negative_iou_thr=args.near_negative_iou_thr,
        train_near_score_only=args.train_near_score_only,
    )
    frozen = verify_frozen_modules(system)
    print(
        f"freeze_check image_trainable={frozen['image_trainable']} "
        f"point_cloud_trainable={frozen['point_cloud_trainable']} "
        f"optimizer_param_count={frozen['optimizer_param_count']}"
    )
    start_epoch, best_loss = load_checkpoint(
        args.resume_ckpt,
        system,
        reset_optimizer=args.reset_optimizer,
        reset_best_loss=args.reset_best_loss,
    ) if args.resume_ckpt else (0, float("inf"))
    train_split = args.train_split_file or args.split_file
    if not args.train_split_file and os.path.basename(train_split) == "val.txt":
        candidate = os.path.join(args.kitti_root, "ImageSets", "train.txt")
        if os.path.isfile(candidate):
            train_split = candidate
    print(f"train_split={train_split}")
    print(
        f"train_with_detector_2d={int(args.train_with_detector_2d)} "
        f"match_2d_iou_thr={args.match_2d_iou_thr:.3f} "
        f"score_positive_radius={args.score_positive_radius:.3f} "
        f"score_raw_miss_radius={args.score_raw_miss_radius:.3f} "
        f"bev_width_meters={args.bev_width_meters:.3f} "
        f"bev_depth_meters={args.bev_depth_meters:.3f} "
        f"bev_resolution={args.bev_resolution:.3f} "
        f"faraway_score_mode={args.faraway_score_mode} "
        f"near_image_score_trainable={int(args.near_image_score_trainable)} "
        f"near_score_loss_weight={args.near_score_loss_weight:.3f} "
        f"near_rank_loss_weight={args.near_rank_loss_weight:.3f} "
        f"near_negative_only_weight={args.near_negative_only_weight:.3f} "
        f"near_score_preserve_weight={args.near_score_preserve_weight:.3f} "
        f"near_iou_thr={args.near_negative_iou_thr:.3f}/{args.near_positive_iou_thr:.3f} "
        f"train_near_score_only={int(args.train_near_score_only)}"
    )
    train_class_ids = parse_train_classes(args.train_classes)
    sample_ids = prepare_train_sample_ids(
        args.kitti_root,
        train_split,
        args.max_samples,
        train_class_ids,
        args.train_faraway_gt_min_depth,
        args.repeat_faraway_samples,
    )
    print(
        f"train_samples={len(sample_ids)} "
        f"train_faraway_gt_min_depth={args.train_faraway_gt_min_depth:.3f} "
        f"repeat_faraway_samples={args.repeat_faraway_samples}"
    )
    last_path = args.save_last_ckpt or args.save_ckpt
    best_path = args.save_best_ckpt
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_loss = 0.0
        grad_steps = 0
        skipped = 0
        log_interval = max(int(args.log_interval), 1)
        epoch_sample_ids = epoch_sample_order(sample_ids, args.seed, epoch)
        total_samples = len(epoch_sample_ids)
        for index, sample_id in enumerate(epoch_sample_ids, start=1):
            batch = make_batch(args.kitti_root, sample_id, args.device, split="training")
            losses = train_step(system, batch)
            grad_step = int(losses.get("grad_step", torch.tensor(1)).item())
            skip = int(losses.get("skipped_no_grad", torch.tensor(0)).item())
            if grad_step:
                epoch_loss += float(losses["loss_total"].item())
                grad_steps += 1
            skipped += skip
            if index == 1 or index % log_interval == 0 or index == total_samples:
                print(
                    f"epoch={epoch + 1} sample_index={index}/{total_samples} sample={sample_id} "
                    f"loss={float(losses['loss_total'].item()):.6f} "
                    f"skip={skip} grad_step={grad_step} "
                    f"num_predictions={int(losses['num_predictions'].item())} "
                    f"{format_loss_metrics(losses)}"
                )
        mean_loss = epoch_loss / max(grad_steps, 1)
        print(f"epoch={epoch + 1} mean_loss={mean_loss:.6f} grad_steps={grad_steps} skipped={skipped}")
        if last_path:
            save_checkpoint(last_path, system, epoch + 1, best_loss)
            print(f"saved_last={last_path}")
        if best_path and grad_steps > 0 and mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(best_path, system, epoch + 1, best_loss)
            print(f"saved_best={best_path} best_loss={best_loss:.6f}")


def run_test(args: argparse.Namespace) -> None:
    system = build_faraway_frustum_system(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        depth_thresholds=parse_depth_thresholds(args.depth_thresholds),
        image_score_thr=args.image_score_thr,
        max_2d_dets=args.max_2d_dets,
        bev_width_meters=args.bev_width_meters,
        bev_depth_meters=args.bev_depth_meters,
        bev_resolution=args.bev_resolution,
        faraway_score_mode=args.faraway_score_mode,
        faraway_image_geometry_alignment=args.faraway_image_geometry_alignment,
        train_class_ids=parse_train_classes(args.train_classes),
        faraway_infer_class_ids=parse_train_classes(args.faraway_infer_classes),
        faraway_score_thr=args.faraway_score_thr,
        faraway_near_suppress_radius=args.faraway_near_suppress_radius,
        criterion_match_radius=args.criterion_match_radius,
        yaw_loss_weight=args.yaw_loss_weight,
        score_loss_weight=args.score_loss_weight,
        train_with_detector_2d=False,
        match_2d_iou_thr=args.match_2d_iou_thr,
        score_positive_radius=args.score_positive_radius,
        score_raw_miss_radius=args.score_raw_miss_radius,
        paper_distance_fusion=args.paper_distance_fusion,
        pedestrian_far_distance=args.pedestrian_far_distance,
        car_far_distance=args.car_far_distance,
        faraway_nms_radius=args.faraway_nms_radius,
        faraway_topk_per_class=args.faraway_topk_per_class,
        car_score_thr=args.car_score_thr,
        point_cloud_model=args.point_cloud_model,
        point_cloud_config=args.point_cloud_config,
        point_cloud_checkpoint=args.point_cloud_ckpt,
        point_cloud_checkpoint_min_bytes=args.point_cloud_ckpt_min_bytes,
        near_image_score_fusion=args.near_image_score_fusion,
        near_image_score_power=args.near_image_score_power,
        near_image_score_min_factor=args.near_image_score_min_factor,
        near_image_score_trainable=args.near_image_score_trainable,
        near_image_score_residual_scale=args.near_image_score_residual_scale,
        near_score_loss_weight=args.near_score_loss_weight,
        near_score_preserve_weight=args.near_score_preserve_weight,
        near_rank_loss_weight=args.near_rank_loss_weight,
        near_rank_margin=args.near_rank_margin,
        near_negative_only_weight=args.near_negative_only_weight,
        near_positive_iou_thr=args.near_positive_iou_thr,
        near_negative_iou_thr=args.near_negative_iou_thr,
        train_near_score_only=False,
    )
    if args.fusion_ckpt:
        load_checkpoint(args.fusion_ckpt, system, reset_optimizer=True, reset_best_loss=False)
    test_split = args.test_split_file or args.split_file
    if not args.test_split_file and os.path.basename(test_split) == "train.txt":
        candidate = os.path.join(args.kitti_root, "ImageSets", "val.txt")
        if os.path.isfile(candidate):
            test_split = candidate
    print(f"test_split={test_split}")
    sample_ids = read_split(test_split, args.max_samples)
    gt_annos = []
    pred_data_dir = args.pred_dir if os.path.basename(args.pred_dir) == "data" else os.path.join(args.pred_dir, "data")
    raw_data_dir = ""
    if args.save_raw_pred_dir:
        raw_data_dir = args.save_raw_pred_dir if os.path.basename(args.save_raw_pred_dir) == "data" else os.path.join(args.save_raw_pred_dir, "data")
    started = time.monotonic()
    total_predictions = 0
    empty_prediction_samples: List[str] = []
    delta_sums = {
        "mean_abs_box_delta": 0.0,
        "max_abs_box_delta": 0.0,
        "mean_abs_score_delta": 0.0,
        "max_abs_score_delta": 0.0,
        "faraway_extra_count": 0.0,
    }
    delta_count = 0
    final_extra_lines = 0
    final_removed_lines = 0
    for index, sample_id in enumerate(sample_ids, start=1):
        batch = make_batch(args.kitti_root, sample_id, args.device, split="training")
        gt_annos.append(batch["gt_anno"])
        path = os.path.join(pred_data_dir, f"{sample_id}.txt")
        if not args.overwrite_pred and os.path.isfile(path) and os.path.getsize(path) > 0:
            result = None
        else:
            result = test_step(system, batch, pred_data_dir)
            path = result["pred_path"]
        raw_result = None
        if raw_data_dir:
            raw_path = os.path.join(raw_data_dir, f"{sample_id}.txt")
            if args.overwrite_pred or not (os.path.isfile(raw_path) and os.path.getsize(raw_path) > 0):
                raw_result = raw_test_step(system, batch, raw_data_dir)
                raw_path = raw_result["pred_path"]
            if args.debug_refine and result is not None and raw_result is not None:
                delta = prediction_delta(result["output"], raw_result["output"])
                delta["faraway_extra_count"] = float(result["output"].get("faraway", {}).get("boxes_3d_camera", torch.zeros((0, 7))).shape[0])
                for key in delta_sums:
                    delta_sums[key] += delta[key]
                delta_count += 1
            if os.path.isfile(path) and os.path.isfile(raw_path):
                with open(path, "r", encoding="utf-8") as pred_handle:
                    pred_lines = set(line.strip() for line in pred_handle if line.strip())
                with open(raw_path, "r", encoding="utf-8") as raw_handle:
                    raw_lines = set(line.strip() for line in raw_handle if line.strip())
                final_extra_lines += len(pred_lines - raw_lines)
                final_removed_lines += len(raw_lines - pred_lines)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        line_count = count_kitti_lines(path)
        total_predictions += line_count
        if line_count == 0:
            empty_prediction_samples.append(sample_id)
        log_interval = max(int(args.log_interval), 1)
        if index == 1 or index % log_interval == 0 or index == len(sample_ids):
            elapsed = time.monotonic() - started
            rate = index / max(elapsed, 1e-6)
            eta = (len(sample_ids) - index) / max(rate, 1e-6)
            print(
                f"test {index}/{len(sample_ids)} pred={path} bytes={size} "
                f"rate={rate:.3f}/s eta_min={eta / 60.0:.1f}",
                flush=True,
            )
    result_files = [os.path.join(pred_data_dir, f"{sample_id}.txt") for sample_id in sample_ids]
    missing = [path for path in result_files if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(f"test did not produce every KITTI file, first={missing[0]}")
    if total_predictions == 0:
        raise RuntimeError("test produced zero total KITTI predictions")
    print(
        f"prediction_summary total_predictions={total_predictions} "
        f"empty_prediction_samples={len(empty_prediction_samples)} "
        f"final_extra_lines={final_extra_lines} "
        f"final_removed_lines={final_removed_lines}"
    )
    if args.debug_refine and delta_count > 0:
        debug = {key: value / delta_count for key, value in delta_sums.items()}
        print(
            "debug_refine "
            + " ".join(f"{key}={value:.6f}" for key, value in debug.items())
        )
    if args.skip_evaluator:
        print("evaluator_skipped=1")
        return
    try:
        eval_result = system.postprocessor.evaluate(pred_data_dir, gt_annos, sample_ids, kitti_root=args.kitti_root)
        print("evaluator_result_begin")
        print(eval_result)
        print("evaluator_result_end")
    except Exception as exc:
        print(f"evaluator_failed: {exc}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--kitti_root", default="/mnt/d/kitti")
    parser.add_argument("--split_file", default="/mnt/d/kitti/ImageSets/val.txt")
    parser.add_argument("--train_split_file", default="")
    parser.add_argument("--test_split_file", default="")
    parser.add_argument("--pred_dir", default="outputs/faraway_frustum/pred")
    parser.add_argument("--save_raw_pred_dir", default="")
    parser.add_argument("--skip_evaluator", action="store_true", help="分片推理时跳过逐分片评测，合并结果后再统一调用 evaluator。")
    parser.add_argument("--save_ckpt", default="")
    parser.add_argument("--save_best_ckpt", default="")
    parser.add_argument("--save_last_ckpt", default="")
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--fusion_ckpt", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--depth_thresholds", default="60,60,75")
    parser.add_argument("--image_score_thr", type=float, default=0.2, help="Mask R-CNN 2D 候选分数阈值。远距离小目标训练/测试可降到 0.05。")
    parser.add_argument("--max_2d_dets", type=int, default=100, help="每帧最多保留的 2D 候选数。")
    parser.add_argument("--bev_width_meters", type=float, default=20.0, help="Faraway 局部 BEV 横向宽度，默认兼顾当前 detector proposal 偏差与局部精度。")
    parser.add_argument("--bev_depth_meters", type=float, default=30.0, help="Faraway 局部 BEV 深度范围，默认兼顾当前 detector proposal 偏差与局部精度。")
    parser.add_argument("--bev_resolution", type=float, default=0.1, help="Faraway 局部 BEV 栅格分辨率，单位米。")
    parser.add_argument("--faraway_score_mode", choices=["image", "product", "quality"], default="image", help="Faraway 最终写出分数来源；默认沿用 2D detector score。")
    parser.add_argument(
        "--faraway_image_geometry_alignment",
        action="store_true",
        help="使用 2D 框启发式覆盖 FF-Net 中心坐标；默认关闭以保留训练得到的框。",
    )
    parser.add_argument("--debug_refine", action="store_true")
    parser.add_argument("--overwrite_pred", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--reset_best_loss", action="store_true")
    parser.add_argument("--train_refine_only", action="store_true")
    parser.add_argument("--train_classes", default="", help="逗号分隔的训练类别名或 ID，例如 Car 或 Pedestrian,Cyclist,Car；空值表示全部类别。")
    parser.add_argument("--faraway_infer_classes", default="", help="逗号分隔的 Faraway 推理类别名或 ID；空值表示不限制类别。")
    parser.add_argument("--faraway_score_thr", type=float, default=0.0)
    parser.add_argument("--faraway_near_suppress_radius", type=float, default=0.0)
    parser.add_argument("--criterion_match_radius", type=float, default=25.0)
    parser.add_argument("--yaw_loss_weight", type=float, default=0.35)
    parser.add_argument("--score_loss_weight", type=float, default=0.5)
    parser.add_argument("--train_with_detector_2d", action="store_true", help="训练时使用冻结 Mask R-CNN 检测框，而不是 GT 2D 框。")
    parser.add_argument("--match_2d_iou_thr", type=float, default=0.5)
    parser.add_argument("--score_positive_radius", type=float, default=4.0, help="Faraway score 正样本允许的 3D 中心误差半径，单位米。")
    parser.add_argument(
        "--score_raw_miss_radius",
        type=float,
        default=0.0,
        help="大于 0 时，Faraway score 只奖励与 raw 3D 检测保持该 BEV 距离以上的补漏候选。",
    )
    parser.add_argument("--paper_distance_fusion", action="store_true", help="使用 Faraway-Frustum 原项目式距离分段融合：近距离保留 raw，远距离保留 Faraway。")
    parser.add_argument("--pedestrian_far_distance", type=float, default=60.0)
    parser.add_argument("--car_far_distance", type=float, default=75.0)
    parser.add_argument("--faraway_nms_radius", type=float, default=0.0, help="Faraway 输出同类 BEV 中心距离去重半径，单位米；0 表示关闭。")
    parser.add_argument("--faraway_topk_per_class", type=int, default=0, help="每帧每类最多写出的 Faraway 候选数；0 表示不限制。")
    parser.add_argument("--car_score_thr", type=float, default=0.0, help="写出 KITTI 结果前过滤低于该分数的 Car 预测；0 表示关闭。")
    parser.add_argument(
        "--point_cloud_model",
        choices=["pv_rcnn_3class", "parta2_car"],
        default="pv_rcnn_3class",
        help="mmdet3d 近距离 detector；parta2_car 为 KITTI Car-only 权重，会把 detector label 0 映射到 KITTI Car。",
    )
    parser.add_argument(
        "--point_cloud_config",
        default="",
        help="显式指定 Faraway 专用 mmdet3d config；空值保持原默认配置。",
    )
    parser.add_argument(
        "--point_cloud_ckpt",
        default="",
        help="显式指定独立训练的点云 detector checkpoint；不存在或残缺时直接报错。",
    )
    parser.add_argument(
        "--point_cloud_ckpt_min_bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="自定义点云 checkpoint 的最小合法字节数。",
    )
    parser.add_argument("--near_image_score_fusion", action="store_true", help="用 near 3D box 投影和同类别 2D 检测证据校准 near detector 分数。")
    parser.add_argument("--near_image_score_power", type=float, default=0.5, help="图像支持分数的指数，越小越温和。")
    parser.add_argument("--near_image_score_min_factor", type=float, default=0.65, help="无图像支持时 near score 的最低保留比例。")
    parser.add_argument("--near_image_score_trainable", action="store_true", help="启用零初始化的可训练 near 图像分数残差。")
    parser.add_argument("--near_image_score_residual_scale", type=float, default=0.5, help="near 分数 logit 残差的最大绝对幅度。")
    parser.add_argument("--near_score_loss_weight", type=float, default=0.0, help="near proposal 3D IoU 质量分数损失权重。")
    parser.add_argument("--near_score_preserve_weight", type=float, default=0.0, help="near 分数残差保持损失权重。")
    parser.add_argument("--near_rank_loss_weight", type=float, default=0.0, help="同帧 near 正负候选 pairwise 排序损失权重。")
    parser.add_argument("--near_rank_margin", type=float, default=0.5, help="near 正负候选 logit 排序间隔。")
    parser.add_argument("--near_negative_only_weight", type=float, default=1.0, help="无正候选帧的 near 负样本损失倍率；0 表示跳过。")
    parser.add_argument("--near_positive_iou_thr", type=float, default=0.70, help="near proposal 正样本 camera 3D IoU 阈值。")
    parser.add_argument("--near_negative_iou_thr", type=float, default=0.30, help="near proposal 负样本 camera 3D IoU 阈值。")
    parser.add_argument("--train_near_score_only", action="store_true", help="冻结 FF-Net，仅训练 near 图像分数校准头。")
    parser.add_argument("--train_faraway_gt_min_depth", type=float, default=0.0, help="大于 0 时，只训练包含该 camera z 深度以上 GT 的样本。")
    parser.add_argument("--repeat_faraway_samples", type=int, default=1, help="启用远距离 GT 过滤后，对过滤样本重复训练的倍数。")
    args = parser.parse_args()
    args.kitti_root = normalize_path(args.kitti_root)
    args.split_file = normalize_path(args.split_file)
    args.train_split_file = normalize_path(args.train_split_file) if args.train_split_file else ""
    args.test_split_file = normalize_path(args.test_split_file) if args.test_split_file else ""
    return args


def parse_depth_thresholds(value: str) -> List[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--depth_thresholds must contain three comma-separated values")
    return vals


def parse_train_classes(value: str) -> List[int]:
    if not value.strip():
        return []
    class_ids: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
            class_id = int(item)
        else:
            if item not in CLASS_TO_ID:
                raise ValueError(f"Unknown class in --train_classes: {item}")
            class_id = CLASS_TO_ID[item]
        if class_id < 0 or class_id >= len(CLASS_TO_ID):
            raise ValueError(f"Class id out of range in --train_classes: {class_id}")
        if class_id not in class_ids:
            class_ids.append(class_id)
    return class_ids


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
