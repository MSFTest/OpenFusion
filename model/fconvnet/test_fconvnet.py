# -*- coding: utf-8 -*-
"""
test_fconvnet.py

F-ConvNet 训练/测试主流程。
支持 KITTI 数据读取、断点续训、checkpoint 保存、KITTI label_2 格式输出。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.fconvnet.initialization.build_fconvnet_system import (
    build_fconvnet_system,
    test_step,
    train_step,
)

LOSS_VERSION = "fconvnet_refinement_residual_quality_v11_iou_axis"


def _trainable_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Persist F-ConvNet itself, not the frozen proposal/baseline detectors."""
    excluded = ("detector2d_backbone.", "point_cloud_backbone.")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    train_refinement_only = bool(getattr(model, "train_refinement_only", False))
    return {
        key: value for key, value in model.state_dict().items()
        if not key.startswith(excluded)
        and (key in trainable or key.startswith("refinement.") or not train_refinement_only)
    }


def _read_split(split_file: str | Path) -> List[str]:
    with Path(split_file).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _read_rgb_detections(
    paths: List[str | Path], score_thr: float = 0.05
) -> Dict[str, Dict[str, torch.Tensor]]:
    grouped: Dict[str, Dict[str, List[Any]]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 7 or int(fields[1]) != 2:
                continue
            score = float(fields[2])
            if score < score_thr:
                continue
            sample_id = Path(fields[0]).stem
            item = grouped.setdefault(sample_id, {"boxes": [], "scores": []})
            item["boxes"].append([float(value) for value in fields[3:7]])
            item["scores"].append(score)
    return {
        sample_id: {
            "boxes": torch.tensor(item["boxes"], dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(item["scores"], dtype=torch.float32),
        }
        for sample_id, item in grouped.items()
    }


def _read_calib(path: Path) -> Dict[str, np.ndarray]:
    calib: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            nums = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
            if key.startswith("P"):
                calib[key] = nums.reshape(3, 4)
            elif key in ("R0_rect", "R_rect"):
                calib["R0_rect"] = nums.reshape(3, 3)
            elif key.startswith("Tr_velo_to_cam") or key.startswith("Tr_velo_cam"):
                calib["Tr_velo_to_cam"] = nums.reshape(3, 4)
    calib.setdefault("R0_rect", np.eye(3, dtype=np.float32))
    calib.setdefault("Tr_velo_to_cam", np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1))
    return calib


def _velodyne_to_rect(points: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    xyz = points[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    cam = xyz_h @ calib["Tr_velo_to_cam"].T
    rect = cam @ calib["R0_rect"].T
    return rect.astype(np.float32)


def _read_image(path: Path) -> torch.Tensor:
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)[:, :, ::-1].copy() / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    except Exception:
        return torch.zeros((3, 375, 1242), dtype=torch.float32)


def _read_labels(path: Path):
    boxes2d = []
    boxes3d = []
    labels = []
    if not path.exists():
        return boxes2d, boxes3d, labels
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] != "Car":
                continue
            bbox = [float(x) for x in parts[4:8]]
            h, w, l = [float(x) for x in parts[8:11]]
            x, y, z = [float(x) for x in parts[11:14]]
            ry = float(parts[14])
            boxes2d.append(bbox)
            boxes3d.append([x, y, z, w, l, h, ry])
            labels.append(0)
    return boxes2d, boxes3d, labels


class KITTIFConvNetDataset(Dataset):
    """
    KITTI F-ConvNet 数据集。

    输出 batch 字段：
        images, points, cam_intrinsic, gt_boxes_2d, gt_boxes_3d, gt_labels_3d, sample_id
    """

    def __init__(
        self,
        kitti_root: str | Path,
        split_file: str | Path,
        max_points: int = 4096,
        car_only_frames: bool = False,
        rgb_detections: Dict[str, Dict[str, torch.Tensor]] | None = None,
        training: bool = False,
    ):
        self.root = Path(kitti_root)
        self.ids = _read_split(split_file)
        self.max_points = int(max_points)
        self.data_dir = self.root / "training"
        self.rgb_detections = rgb_detections
        self.training = training
        if car_only_frames:
            label_dir = self.data_dir / "label_2"
            self.ids = [
                sample_id
                for sample_id in self.ids
                if any(
                    line.startswith("Car ")
                    for line in (label_dir / f"{sample_id}.txt").read_text(encoding="utf-8").splitlines()
                )
            ]
        if training and rgb_detections is not None:
            self.ids = [sample_id for sample_id in self.ids if sample_id in rgb_detections]

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _subsample_points(points: np.ndarray, max_points: int, sample_id: str, training: bool) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        if training:
            choice = np.random.choice(points.shape[0], max_points, replace=False)
        else:
            seed = zlib.crc32(sample_id.encode("utf-8"))
            choice = np.random.default_rng(seed).choice(points.shape[0], max_points, replace=False)
        return points[choice]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.ids[idx]
        image = _read_image(self.data_dir / "image_2" / f"{sample_id}.png")
        calib = _read_calib(self.data_dir / "calib" / f"{sample_id}.txt")
        velodyne_path = self.data_dir / "velodyne" / f"{sample_id}.bin"
        points_raw = np.fromfile(str(velodyne_path), dtype=np.float32).reshape(-1, 4)
        points_rect = _velodyne_to_rect(points_raw, calib)
        points_rect = points_rect[points_rect[:, 2] > 0.1]
        boxes2d, boxes3d, labels = _read_labels(self.data_dir / "label_2" / f"{sample_id}.txt")
        detection = self.rgb_detections.get(sample_id) if self.rgb_detections is not None else None
        crop_boxes = detection["boxes"].numpy() if detection is not None else np.asarray(boxes2d, dtype=np.float32)
        if len(crop_boxes) > 0 and points_rect.shape[0] > 0:
            p2 = calib["P2"]
            homogeneous = np.concatenate(
                [points_rect, np.ones((points_rect.shape[0], 1), dtype=np.float32)], axis=1
            )
            projected = homogeneous @ p2.T
            uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
            in_any_box = np.zeros((points_rect.shape[0],), dtype=bool)
            for x1, y1, x2, y2 in crop_boxes:
                in_any_box |= (
                    (uv[:, 0] >= x1) & (uv[:, 0] <= x2)
                    & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
                )
            points_rect = points_rect[in_any_box]
        points_rect = self._subsample_points(points_rect, self.max_points, sample_id, self.training)

        H, W = image.shape[1], image.shape[2]
        P2 = calib.get("P2", np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1))
        rect4 = np.eye(4, dtype=np.float32)
        rect4[:3, :3] = calib["R0_rect"]
        velo2cam4 = np.eye(4, dtype=np.float32)
        velo2cam4[:3, :4] = calib["Tr_velo_to_cam"]
        lidar2cam = rect4 @ velo2cam4
        sample = {
            "sample_id": sample_id,
            "pcd_path": str(velodyne_path),
            "velodyne_path": str(velodyne_path),
            "images": image,
            "points": torch.from_numpy(points_rect).float(),
            "cam_intrinsic": torch.from_numpy(P2[:, :3]).float(),
            "P2": torch.from_numpy(P2).float(),
            "lidar2cam": torch.from_numpy(lidar2cam).float(),
            "gt_boxes_2d": torch.tensor(boxes2d, dtype=torch.float32).reshape(-1, 4),
            "gt_boxes_3d": torch.tensor(boxes3d, dtype=torch.float32).reshape(-1, 7),
            "gt_labels_3d": torch.tensor(labels, dtype=torch.long),
            "image_shape": (H, W),
        }
        if self.rgb_detections is not None:
            if detection is None:
                sample["det_boxes_2d"] = torch.zeros((0, 4), dtype=torch.float32)
                sample["det_scores_2d"] = torch.zeros((0,), dtype=torch.float32)
            else:
                sample["det_boxes_2d"] = detection["boxes"]
                sample["det_scores_2d"] = detection["scores"]
        return sample


def _pad_tensor_list(items: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(x.shape[0] for x in items)
    tail = items[0].shape[1:]
    out = items[0].new_full((len(items), max_len) + tail, pad_value)
    for i, x in enumerate(items):
        out[i, : x.shape[0]] = x
    return out


def _pad_images(images: List[torch.Tensor]) -> torch.Tensor:
    max_h = max(image.shape[1] for image in images)
    max_w = max(image.shape[2] for image in images)
    out = images[0].new_zeros((len(images), images[0].shape[0], max_h, max_w))
    for idx, image in enumerate(images):
        out[idx, :, :image.shape[1], :image.shape[2]] = image
    return out


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {
        "sample_id": [s["sample_id"] for s in samples],
        "pcd_path": [s["pcd_path"] for s in samples],
        "velodyne_path": [s["velodyne_path"] for s in samples],
        "images": _pad_images([s["images"] for s in samples]),
        "points": _pad_tensor_list([s["points"] for s in samples]),
        "cam_intrinsic": torch.stack([s["cam_intrinsic"] for s in samples], dim=0),
        "P2": torch.stack([s["P2"] for s in samples], dim=0),
        "lidar2cam": torch.stack([s["lidar2cam"] for s in samples], dim=0),
        "gt_boxes_2d": _pad_tensor_list([s["gt_boxes_2d"] for s in samples]),
        "gt_boxes_3d": _pad_tensor_list([s["gt_boxes_3d"] for s in samples]),
        "gt_labels_3d": _pad_tensor_list([s["gt_labels_3d"].view(-1, 1).float() for s in samples]).squeeze(-1).long(),
        "image_shape": (
            max(s["image_shape"][0] for s in samples),
            max(s["image_shape"][1] for s in samples),
        ),
    }
    if "det_boxes_2d" in samples[0]:
        batch["det_boxes_2d"] = _pad_tensor_list([s["det_boxes_2d"] for s in samples])
        batch["det_scores_2d"] = _pad_tensor_list(
            [s["det_scores_2d"].view(-1, 1) for s in samples]
        ).squeeze(-1)
    return batch


def save_checkpoint(path: str | Path, system, epoch: int, best_loss: float):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "loss_version": LOSS_VERSION,
                "refine_center_loss_weight": getattr(
                    getattr(system, "criterion", None), "refine_center_loss_weight", 1.0
                ),
                "model": _trainable_model_state(system.model),
                "optimizer": system.optimizer.state_dict(),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_checkpoint(path: str | Path, system, load_optimizer: bool = True):
    ckpt = torch.load(path, map_location=system.device)
    incompatible = system.model.load_state_dict(ckpt["model"], strict=False)
    relevant_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(("detector2d_backbone.", "point_cloud_backbone."))
    ]
    print(
        f"loaded_checkpoint={path} epoch={int(ckpt.get('epoch', 0))} "
        f"loss_version={ckpt.get('loss_version', 'legacy')} "
        f"missing={len(relevant_missing)} refine_missing={sum(key.startswith('refinement.') for key in relevant_missing)} "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    if load_optimizer and "optimizer" in ckpt:
        system.optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_loss", float("inf")))


def run_train(args):
    split_file = args.train_split_file or args.split_file or Path(args.kitti_root) / "ImageSets" / "train.txt"
    rgb_detections = (
        _read_rgb_detections(args.rgb_detection_files, score_thr=args.proposal_score_thr)
        if args.rgb_detection_files else None
    )
    dataset = KITTIFConvNetDataset(
        args.kitti_root,
        split_file,
        max_points=args.max_points,
        car_only_frames=True,
        rgb_detections=rgb_detections,
        training=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    system = build_fconvnet_system(
        image_cfg=args.image_cfg,
        image_ckpt=args.image_ckpt,
        point_cloud_cfg=args.point_cloud_cfg,
        point_cloud_ckpt=args.point_cloud_ckpt,
        device=args.device,
        lr=args.lr,
        use_gt_2d=rgb_detections is None,
        use_precomputed_2d=rgb_detections is not None,
        enable_refinement=args.train_refinement,
        train_refinement_only=args.train_refinement,
        refine_residual_scale=args.refine_residual_scale,
        refine_residual_gate_score=args.refine_residual_gate_score,
        refine_center_loss_weight=args.refine_center_loss_weight,
        refine_min_iou=args.refine_min_iou,
    )
    start_epoch, best_loss = 0, float("inf")
    if args.train_refinement:
        if not args.stage1_ckpt:
            raise ValueError("--train_refinement requires --stage1_ckpt")
        load_checkpoint(args.stage1_ckpt, system, load_optimizer=False)
    if args.resume_ckpt:
        start_epoch, best_loss = load_checkpoint(
            args.resume_ckpt, system, load_optimizer=not args.reset_optimizer
        )
        if args.reset_best_loss:
            best_loss = float("inf")

    save_last_path = Path(args.save_last_ckpt or args.save_ckpt)
    save_best_path = Path(args.save_best_ckpt)
    for epoch in range(start_epoch, args.epochs):
        epoch_lr = args.lr * (args.lr_decay_gamma if epoch >= args.lr_decay_epoch else 1.0)
        for group in system.optimizer.param_groups:
            group["lr"] = epoch_lr
        print(f"[train] epoch={epoch} lr={epoch_lr:.8f}")
        running = 0.0
        grad_steps = 0
        skipped_steps = 0
        for step, batch in enumerate(loader):
            batch["use_gt_2d"] = rgb_detections is None
            loss_dict = train_step(system, batch)
            loss_val = float(loss_dict["loss"].detach().cpu())
            skipped = int(loss_dict.get("skipped_no_grad", 0))
            if skipped:
                skipped_steps += 1
            else:
                running += loss_val
                grad_steps += 1
            if step % args.log_interval == 0:
                detail = ""
                if args.debug_refine:
                    detail = (
                        f" cls={float(loss_dict.get('cls_loss', 0)):.6f}"
                        f" reg={float(loss_dict.get('reg_loss', 0)):.6f}"
                        f" pos={int(loss_dict.get('num_pos', 0))}"
                        f" neg={int(loss_dict.get('num_neg', 0))}"
                        f" ign={int(loss_dict.get('num_ign', 0))}"
                        f" quality={int(loss_dict.get('num_quality_proposals', 0))}"
                    )
                print(
                    f"[train] epoch={epoch} step={step} loss={loss_val:.6f} "
                    f"skip={skipped}{detail}"
                )
            if args.max_steps > 0 and step + 1 >= args.max_steps:
                break
        if grad_steps == 0:
            raise RuntimeError("epoch completed without any F-ConvNet gradient steps")
        epoch_loss = running / grad_steps
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(save_best_path, system, epoch + 1, best_loss)
        save_checkpoint(save_last_path, system, epoch + 1, best_loss)
        print(
            f"[train] epoch={epoch} avg_loss={epoch_loss:.6f} "
            f"grad_steps={grad_steps} skipped_steps={skipped_steps}"
        )


def run_test(args):
    split_file = args.test_split_file or args.split_file or Path(args.kitti_root) / "ImageSets" / "val.txt"
    rgb_detections = (
        _read_rgb_detections(args.rgb_detection_files, score_thr=args.proposal_score_thr)
        if args.rgb_detection_files
        else None
    )
    dataset = KITTIFConvNetDataset(
        args.kitti_root,
        split_file,
        max_points=args.max_points,
        rgb_detections=rgb_detections,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    system = build_fconvnet_system(
        image_cfg=args.image_cfg,
        image_ckpt=args.image_ckpt,
        point_cloud_cfg=args.point_cloud_cfg,
        point_cloud_ckpt=args.point_cloud_ckpt,
        device=args.device,
        lr=args.lr,
        use_precomputed_2d=rgb_detections is not None,
        enable_raw_detector=bool(args.save_raw_pred_dir) and not args.enable_refinement,
        enable_refinement=args.enable_refinement,
        refine_residual_scale=args.refine_residual_scale,
        refine_residual_gate_score=args.refine_residual_gate_score,
        score_thr=args.score_thr,
        nms_iou_thr=args.nms_iou_thr,
        refine_score_weight=args.refine_score_weight,
    )
    if args.enable_refinement:
        if not args.stage1_ckpt:
            raise ValueError("--enable_refinement requires --stage1_ckpt")
        load_checkpoint(args.stage1_ckpt, system, load_optimizer=False)
    if args.fusion_ckpt:
        load_checkpoint(args.fusion_ckpt, system, load_optimizer=False)

    pred_dir = Path(args.pred_dir)
    raw_pred_dir = Path(args.save_raw_pred_dir) if args.save_raw_pred_dir else None
    total_predictions = 0
    empty_predictions = 0
    for step, batch in enumerate(loader):
        results = test_step(
            system, batch, include_raw=raw_pred_dir is not None,
            refine_batch_stats=args.refine_bn_batch_stats,
        )
        raw_results = None
        if raw_pred_dir is not None:
            raw_results = results["raw"]
            results = results["fconvnet"]
        sample_ids = batch["sample_id"]
        # 当前实现对 batch 合并结果统一输出到第一张图；batch_size=1 是官方验证推荐路径。
        system.postprocessor.write_kitti(results, pred_dir, sample_ids[0])
        count = int(results["boxes_3d"].shape[0])
        total_predictions += count
        empty_predictions += int(count == 0)
        if raw_pred_dir is not None and raw_results is not None:
            raw_sample = raw_results.get("sample_results", [raw_results])[0]
            system.postprocessor.write_kitti(raw_sample, raw_pred_dir, sample_ids[0])
        if step % args.log_interval == 0:
            print(f"[test] step={step} sample={sample_ids[0]} predictions={count}")
        if args.max_steps > 0 and step + 1 >= args.max_steps:
            break
    print(
        f"total_predictions={total_predictions} empty_prediction_samples={empty_predictions}"
    )
    if total_predictions == 0:
        raise RuntimeError("F-ConvNet produced no predictions for the entire test split")
    if args.eval:
        result_root = pred_dir.parent if pred_dir.name == "data" else pred_dir
        if not (result_root / "data").is_dir():
            raise ValueError("--pred_dir must point to a directory named 'data' for native KITTI evaluation")
        evaluator = Path(args.native_eval_root) / "build" / "evaluate_object_3d_offline"
        completed = subprocess.run(
            [str(evaluator), str(Path(args.kitti_root) / "training" / "label_2"), str(result_root)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)


def build_argparser():
    parser = argparse.ArgumentParser(description="F-ConvNet train/test")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--kitti_root", default="/mnt/d/kitti")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--train_split_file", default=None)
    parser.add_argument("--test_split_file", default=None)
    parser.add_argument("--pred_dir", default="pred_kitti/data")
    parser.add_argument("--image_cfg", default=None)
    parser.add_argument("--image_ckpt", default=None)
    parser.add_argument("--rgb_detection_files", nargs="*", default=None)
    parser.add_argument("--proposal_score_thr", type=float, default=0.05)
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--nms_iou_thr", type=float, default=0.1)
    parser.add_argument("--point_cloud_cfg", default=None)
    parser.add_argument("--point_cloud_ckpt", default=None)
    parser.add_argument("--save_ckpt", default="/tmp/test_ckpt.pth")
    parser.add_argument("--save_best_ckpt", default="checkpoints/fconvnet_best.pth")
    parser.add_argument("--save_last_ckpt", default="checkpoints/fconvnet_last.pth")
    parser.add_argument("--save_raw_pred_dir", default=None)
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument("--fusion_ckpt", default=None)
    parser.add_argument("--stage1_ckpt", default=None)
    parser.add_argument("--train_refinement", action="store_true")
    parser.add_argument("--enable_refinement", action="store_true")
    parser.add_argument("--refine_residual_scale", type=float, default=1.0)
    parser.add_argument("--refine_residual_gate_score", type=float, default=float("inf"))
    parser.add_argument("--refine_score_weight", type=float, default=0.0)
    parser.add_argument("--refine_bn_batch_stats", action="store_true")
    parser.add_argument("--refine_center_loss_weight", type=float, default=1.0)
    parser.add_argument("--refine_min_iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_epoch", type=int, default=20)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--native_eval_root", default="/root/projects/FusionRepo/kitti_native_evaluation")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--reset_best_loss", action="store_true")
    parser.add_argument("--debug_refine", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
