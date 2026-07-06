"""Real KITTI train/test entry point for the FusionRepo BEVFusion model."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.bevfusion.initialization.build_bevfusion_system import (
    build_bevfusion_system,
    test_step,
    train_step,
)


KITTI_CLASSES = ("Pedestrian", "Cyclist", "Car")
CLASS_TO_INDEX = {name: index for index, name in enumerate(KITTI_CLASSES)}


def read_split(split_file: str) -> list[str]:
    samples = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError(f"split file is empty: {split_file}")
    return samples


def read_calibration(path: Path) -> dict[str, torch.Tensor]:
    raw = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        raw[key] = np.asarray([float(value) for value in values.split()], dtype=np.float32)
    return {
        "P2": torch.from_numpy(raw["P2"].reshape(3, 4)),
        "R0_rect": torch.from_numpy(raw["R0_rect"].reshape(3, 3)),
        "Tr_velo2cam": torch.from_numpy(
            raw.get("Tr_velo_to_cam", raw.get("Tr_velo2cam")).reshape(3, 4)
        ),
    }


def _to44(matrix: torch.Tensor) -> torch.Tensor:
    output = torch.eye(4, dtype=matrix.dtype)
    output[: matrix.shape[0], : matrix.shape[1]] = matrix
    return output


def read_ground_truth(path: Path, calibration: dict) -> tuple[torch.Tensor, torch.Tensor]:
    boxes, labels = [], []
    camera_to_lidar = torch.linalg.inv(
        _to44(calibration["R0_rect"]) @ _to44(calibration["Tr_velo2cam"])
    )
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] not in CLASS_TO_INDEX:
            continue
        height, width, length = map(float, fields[8:11])
        x, y, z = map(float, fields[11:14])
        rotation_y = float(fields[14])
        camera_location = torch.tensor([x, y, z, 1.0], dtype=torch.float32)
        lidar_location = (camera_to_lidar @ camera_location)[:3]
        boxes.append(
            [
                float(lidar_location[0]),
                float(lidar_location[1]),
                float(lidar_location[2]),
                length,
                width,
                height,
                -rotation_y - math.pi / 2.0,
            ]
        )
        labels.append(CLASS_TO_INDEX[fields[0]])
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7),
        torch.tensor(labels, dtype=torch.long),
    )


def build_batch(kitti_root: str, sample_id: str) -> dict:
    root = Path(kitti_root) / "training"
    image_path = root / "image_2" / f"{sample_id}.png"
    point_path = root / "velodyne" / f"{sample_id}.bin"
    calibration = read_calibration(root / "calib" / f"{sample_id}.txt")
    image_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32).copy()
    image = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    height, width = image.shape[-2:]
    padded_height = int(math.ceil(height / 32) * 32)
    padded_width = int(math.ceil(width / 32) * 32)
    image = F.pad(image, (0, padded_width - width, 0, padded_height - height))
    points = torch.from_numpy(
        np.fromfile(str(point_path), dtype=np.float32).reshape(-1, 4)
    )
    boxes, labels = read_ground_truth(
        root / "label_2" / f"{sample_id}.txt", calibration
    )
    return {
        "sample_ids": [sample_id],
        "images": image.unsqueeze(0),
        "points": [points],
        "point_paths": [str(point_path)],
        "calibs": [calibration],
        "image_shapes": [(height, width)],
        "padded_image_shapes": [(padded_height, padded_width)],
        "gt_boxes_3d": [boxes],
        "gt_labels_3d": [labels],
    }


def annotation_to_lines(annotation: dict) -> list[str]:
    lines = []
    for index, name in enumerate(annotation["name"]):
        bbox = annotation["bbox"][index]
        dimensions = annotation["dimensions"][index]
        location = annotation["location"][index]
        lines.append(
            f"{name} {annotation['truncated'][index]:.2f} {int(annotation['occluded'][index])} "
            f"{annotation['alpha'][index]:.6f} "
            f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
            f"{dimensions[0]:.4f} {dimensions[1]:.4f} {dimensions[2]:.4f} "
            f"{location[0]:.4f} {location[1]:.4f} {location[2]:.4f} "
            f"{annotation['rotation_y'][index]:.6f} {annotation['score'][index]:.6f}"
        )
    return lines


def write_prediction(pred_dir: str, sample_id: str, annotation: dict) -> Path:
    directory = Path(pred_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lines = annotation_to_lines(annotation)
    if not lines:
        raise RuntimeError(f"real detector returned no predictions for KITTI sample {sample_id}")
    path = directory / f"{sample_id}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def filter_invalid_kitti_predictions(paths: list[Path]) -> int:
    """Remove camera-invisible boxes using the official KITTI validity rule."""
    removed = 0
    for path in paths:
        valid_lines = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 16:
                continue
            left, top, right, bottom = map(float, fields[4:8])
            camera_z = float(fields[13])
            if right <= left or bottom <= top or camera_z <= 0:
                removed += 1
                continue
            valid_lines.append(line)
        if not valid_lines:
            raise RuntimeError(f"all detector predictions are camera-invisible: {path}")
        path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    return removed


def read_kitti_annotation(path: Path, prediction: bool = False) -> dict:
    fields = {
        "name": [], "truncated": [], "occluded": [], "alpha": [], "bbox": [],
        "dimensions": [], "location": [], "rotation_y": [], "score": []
    }
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) < 15:
            continue
        fields["name"].append(values[0])
        fields["truncated"].append(float(values[1]))
        fields["occluded"].append(int(values[2]))
        fields["alpha"].append(float(values[3]))
        fields["bbox"].append([float(value) for value in values[4:8]])
        fields["dimensions"].append([float(value) for value in values[8:11]])
        fields["location"].append([float(value) for value in values[11:14]])
        fields["rotation_y"].append(float(values[14]))
        fields["score"].append(float(values[15]) if len(values) > 15 else 0.0)
    count = len(fields["name"])
    annotation = {
        "name": np.asarray(fields["name"], dtype=object),
        "truncated": np.asarray(fields["truncated"], dtype=np.float32),
        "occluded": np.asarray(fields["occluded"], dtype=np.int64),
        "alpha": np.asarray(fields["alpha"], dtype=np.float32),
        "bbox": np.asarray(fields["bbox"], dtype=np.float32).reshape(count, 4),
        "dimensions": np.asarray(fields["dimensions"], dtype=np.float32).reshape(count, 3),
        "location": np.asarray(fields["location"], dtype=np.float32).reshape(count, 3),
        "rotation_y": np.asarray(fields["rotation_y"], dtype=np.float32),
    }
    if prediction:
        annotation["score"] = np.asarray(fields["score"], dtype=np.float32)
    return annotation


def evaluate_kitti(kitti_root: str, sample_ids: list[str], pred_dir: str) -> dict:
    # This environment's numba/CUDA pair segfaults while compiling rotate_iou.
    # CUDA simulation executes the same official evaluator deterministically.
    os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")
    from mmdet3d.evaluation.functional.kitti_utils.eval import kitti_eval

    gt = [
        read_kitti_annotation(
            Path(kitti_root) / "training" / "label_2" / f"{sample_id}.txt"
        )
        for sample_id in sample_ids
    ]
    predictions = [
        read_kitti_annotation(Path(pred_dir) / f"{sample_id}.txt", prediction=True)
        for sample_id in sample_ids
    ]
    report, metrics = kitti_eval(
        gt_annos=gt,
        dt_annos=predictions,
        current_classes=list(KITTI_CLASSES),
        eval_types=["bbox", "bev", "3d"],
    )
    return {"report": report, "metrics": metrics}


def parse_native_ap40(result_root: Path, class_names=KITTI_CLASSES) -> dict:
    """Parse the native evaluator's 41-point precision rows into AP40."""
    metrics = {}
    for class_name in class_names:
        class_key = class_name.lower()
        output = {}
        for metric_name, suffix in (
            ("bbox_ap40", "detection.txt"),
            ("3d_ap40", "detection_ground.txt"),
        ):
            path = result_root / f"stats_{class_key}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(f"native evaluator stats missing: {path}")
            rows = []
            for line in path.read_text().splitlines()[:3]:
                precision = [float(value) for value in line.split()]
                if len(precision) < 41:
                    raise ValueError(f"invalid native precision row in {path}")
                rows.append(round(sum(precision[1:41]) / 40.0 * 100.0, 4))
            if len(rows) != 3:
                raise ValueError(f"native evaluator did not emit three difficulties: {path}")
            output[metric_name] = rows
        metrics[class_name] = output
    return metrics


def evaluate_kitti_native(
    kitti_root: str,
    pred_dir: str,
    native_eval_root: str = "kitti_native_evaluation",
) -> dict:
    """Run the downloaded C++ KITTI evaluator and return AP40 metrics."""
    prediction_data = Path(pred_dir).resolve()
    result_root = prediction_data.parent if prediction_data.name == "data" else prediction_data
    expected_data = result_root / "data"
    if prediction_data != expected_data:
        raise ValueError("native KITTI evaluator requires pred_dir to end with '/data'")
    evaluator = Path(native_eval_root).resolve() / "build/evaluate_object_3d_offline"
    if not evaluator.is_file():
        raise FileNotFoundError(f"native KITTI evaluator binary not found: {evaluator}")
    ground_truth = Path(kitti_root).resolve() / "training/label_2"
    completed = subprocess.run(
        [str(evaluator), str(ground_truth), str(result_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "metrics": parse_native_ap40(result_root),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def save_checkpoint(path: str, system, epoch: int) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model": system.model.state_dict(), "optimizer": system.optimizer.state_dict()},
        destination,
    )


def resume_checkpoint(path: str, system) -> int:
    checkpoint = torch.load(path, map_location=system.device)
    system.model.load_state_dict(checkpoint["model"], strict=False)
    if "optimizer" in checkpoint:
        system.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def _build_system(args):
    return build_bevfusion_system(
        device=args.device,
        fusion_checkpoint=args.fusion_ckpt or None,
        lr=args.lr,
    )


def run_train(args) -> None:
    system = _build_system(args)
    start_epoch = resume_checkpoint(args.resume_ckpt, system) if args.resume_ckpt else 0
    sample_ids = read_split(args.split_file)
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]
    log_interval = max(int(args.log_interval), 1)
    for epoch in range(start_epoch, args.epochs):
        running_loss = 0.0
        running_box = 0.0
        running_score = 0.0
        running_count = 0
        for index, sample_id in enumerate(sample_ids, start=1):
            losses = train_step(system, build_batch(args.kitti_root, sample_id))
            running_loss += float(losses["loss"].item())
            running_box += float(losses["loss_box"].item())
            running_score += float(losses["loss_score"].item())
            running_count += 1
            should_log = running_count == log_interval or index == len(sample_ids)
            if not should_log:
                continue
            print(
                f"epoch={epoch + 1} samples={index - running_count + 1}-{index}/{len(sample_ids)} "
                f"avg_loss={running_loss / running_count:.6f} "
                f"avg_box={running_box / running_count:.6f} "
                f"avg_score={running_score / running_count:.6f}",
                flush=True,
            )
            running_loss = 0.0
            running_box = 0.0
            running_score = 0.0
            running_count = 0
        save_checkpoint(args.save_ckpt, system, epoch + 1)


def run_test(args) -> None:
    system = _build_system(args)
    if args.resume_ckpt:
        resume_checkpoint(args.resume_ckpt, system)
    sample_ids = read_split(args.split_file)
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]
    started = time.monotonic()
    completed_count = 0
    for index, sample_id in enumerate(sample_ids, start=1):
        existing = Path(args.pred_dir) / f"{sample_id}.txt"
        if existing.is_file() and existing.stat().st_size > 0:
            completed_count += 1
            continue
        output = test_step(system, build_batch(args.kitti_root, sample_id))
        write_prediction(args.pred_dir, sample_id, output["annotations"][0])
        completed_count += 1
        if completed_count % 20 == 0 or completed_count == len(sample_ids):
            elapsed = time.monotonic() - started
            rate = completed_count / max(elapsed, 1e-6)
            remaining = (len(sample_ids) - completed_count) / max(rate, 1e-6)
            print(
                f"test {completed_count}/{len(sample_ids)} "
                f"rate={rate:.2f} samples/s eta={remaining / 60.0:.1f} min",
                flush=True,
            )
    result_files = [Path(args.pred_dir) / f"{sample_id}.txt" for sample_id in sample_ids]
    if any(not path.is_file() or path.stat().st_size == 0 for path in result_files):
        raise RuntimeError("full KITTI test did not produce every non-empty result file")
    removed = filter_invalid_kitti_predictions(result_files)
    print(f"filtered_camera_invisible_predictions={removed}")
    if args.evaluator == "native":
        evaluation = evaluate_kitti_native(
            args.kitti_root, args.pred_dir, args.native_eval_root
        )
        for class_name, values in evaluation["metrics"].items():
            print(
                f"{class_name}: bbox_AP40={values['bbox_ap40']} "
                f"3d_AP40={values['3d_ap40']}"
            )
    elif args.evaluator == "python":
        evaluation = evaluate_kitti(args.kitti_root, sample_ids, args.pred_dir)
        print(evaluation["report"])


def parse_args():
    parser = argparse.ArgumentParser(description="FusionRepo BEVFusion KITTI runner")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--kitti_root", default="/mnt/d/kitti")
    parser.add_argument("--split_file", default="")
    parser.add_argument("--pred_dir", default="results/bevfusion_kitti/data")
    parser.add_argument("--save_ckpt", default="checkpoints/bevfusion_fusion_last.pth")
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--fusion_ckpt", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--evaluator", choices=["native", "python", "none"], default="native")
    parser.add_argument("--native_eval_root", default="kitti_native_evaluation")
    args = parser.parse_args()
    if not args.split_file:
        split = "train.txt" if args.mode == "train" else "val.txt"
        args.split_file = str(Path(args.kitti_root) / "ImageSets" / split)
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "train":
        run_train(arguments)
    else:
        run_test(arguments)
