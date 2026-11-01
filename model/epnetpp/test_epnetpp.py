# -*- coding: utf-8 -*-
"""EPNet++ train/test entrypoint."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.epnetpp.initialization.build_epnetpp_system import (  # noqa: E402
    MMDET3D_CHECKPOINT,
    MMDET3D_CONFIG,
    MMDET3D_MODEL_NAME,
    MMDET_CHECKPOINT,
    MMDET_CONFIG,
    MMDET_MODEL_NAME,
    LOSS_VERSION,
    build_epnetpp_system,
    test_step,
    train_step,
)


def _resolve_kitti_root(root: str) -> Path:
    base = Path(root)
    for cand in (base, base / "object"):
        if (cand / "training").exists() or (cand / "testing").exists():
            return cand
    return base


def _read_split(split_file: Optional[str], root: Path, mode: str) -> List[str]:
    if split_file:
        path = Path(split_file)
    else:
        split = "train" if mode == "train" else "val"
        path = root.parent / "ImageSets" / f"{split}.txt"
        if not path.exists():
            path = root / "ImageSets" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _find_file(root: Path, sample_id: str, rels: Iterable[str]) -> Path:
    for rel in rels:
        path = root / rel.format(id=sample_id)
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find sample {sample_id} in {root}: {list(rels)}")


def _read_calib(path: Path) -> Dict[str, torch.Tensor]:
    data: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        arr = np.array([float(x) for x in values.strip().split()], dtype=np.float32)
        data[key] = arr
    p2 = data.get("P2")
    r0 = data.get("R0_rect", data.get("R_rect"))
    tr = data.get("Tr_velo_to_cam", data.get("Tr_velo2cam"))
    if p2 is None or r0 is None or tr is None:
        raise ValueError(f"KITTI calibration is incomplete: {path}")
    return {
        "P2": torch.from_numpy(p2.reshape(3, 4)),
        "R0_rect": torch.from_numpy(r0.reshape(3, 3)),
        "Tr_velo2cam": torch.from_numpy(tr.reshape(3, 4)),
    }


def _load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    arr = arr[:, :, ::-1].copy()
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _load_points(path: Path, max_points: int = 16384, random_sample: bool = False) -> torch.Tensor:
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    if points.shape[0] > max_points:
        if random_sample:
            idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        else:
            idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]
    return torch.from_numpy(points).float().unsqueeze(0)


def _projection_mats(calib: Dict[str, torch.Tensor]):
    r0 = calib["R0_rect"]
    tr = calib["Tr_velo2cam"]
    return r0 @ tr[:, :3], r0 @ tr[:, 3], calib["P2"][:, :3]


def _to_4x4(mat: torch.Tensor) -> torch.Tensor:
    out = mat.new_zeros((4, 4))
    out[3, 3] = 1.0
    out[: mat.shape[0], : mat.shape[1]] = mat
    return out


def _camera_box_to_lidar(
    h: float,
    w: float,
    l: float,
    x: float,
    y: float,
    z: float,
    ry: float,
    calib: Dict[str, torch.Tensor],
) -> List[float]:
    """将 KITTI label_2 camera 坐标框转换为 LiDAR bottom-center 框。"""
    r0 = _to_4x4(calib["R0_rect"].float())
    tr = _to_4x4(calib["Tr_velo2cam"].float())
    rect_to_lidar = torch.inverse(r0 @ tr)
    center_cam = torch.tensor([x, y, z, 1.0], dtype=torch.float32)
    center_lidar = (rect_to_lidar @ center_cam)[:3]
    yaw_lidar = -float(ry) - np.pi / 2.0
    return [
        float(center_lidar[0]),
        float(center_lidar[1]),
        float(center_lidar[2]),
        float(l),
        float(w),
        float(h),
        float(yaw_lidar),
    ]


def _read_label_boxes(path: Path, calib: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not path.exists():
        return torch.zeros((0, 7), dtype=torch.float32)
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] == "DontCare":
            continue
        h, w, l = [float(x) for x in parts[8:11]]
        x, y, z = [float(x) for x in parts[11:14]]
        ry = float(parts[14])
        boxes.append(_camera_box_to_lidar(h, w, l, x, y, z, ry, calib))
    if not boxes:
        return torch.zeros((0, 7), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _build_batch(root: Path, sample_id: str, mode: str, max_points: int = 16384) -> Dict[str, object]:
    image_path = _find_file(root, sample_id, ["training/image_2/{id}.png", "testing/image_2/{id}.png"])
    velo_path = _find_file(root, sample_id, ["training/velodyne/{id}.bin", "testing/velodyne/{id}.bin"])
    calib_path = _find_file(root, sample_id, ["training/calib/{id}.txt", "testing/calib/{id}.txt"])
    calib = _read_calib(calib_path)
    r, t, k = _projection_mats(calib)
    batch: Dict[str, object] = {
        "sample_id": sample_id,
        "pcd_path": str(velo_path),
        "images": _load_image(image_path),
        "points": _load_points(velo_path, max_points=max_points, random_sample=(mode == "train")),
        "calib": calib,
        "lidar_to_cam_r": r.unsqueeze(0),
        "lidar_to_cam_t": t.unsqueeze(0),
        "cam_intrinsic": k.unsqueeze(0),
    }
    if mode == "train":
        label_path = root / "training" / "label_2" / f"{sample_id}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Training label_2 not found: {label_path}")
        batch["gt_boxes_3d"] = _read_label_boxes(label_path, calib)
    return batch


def evaluate_predictions(root: Path, pred_dir: Path):
    """调用 KITTI native 官方评测程序。"""
    label_dir = root / "training" / "label_2"
    if not label_dir.exists():
        raise RuntimeError(f"KITTI label_2 directory not found: {label_dir}")
    native_eval = None
    for cand in (
        REPO_ROOT / "kitti_native_evaluation" / "build" / "evaluate_object_3d_offline",
        REPO_ROOT / "kitti_native_evaluation" / "evaluate_object_3d_offline",
        Path("/mnt/d/kitti_native_evaluation/build/evaluate_object_3d_offline"),
        Path("/mnt/d/kitti_native_evaluation/evaluate_object_3d_offline"),
    ):
        if cand.exists():
            native_eval = cand
            break
    if native_eval is None:
        raise RuntimeError("KITTI native evaluator not found in repo or /mnt/d/kitti_native_evaluation.")
    proc = subprocess.run(
        [str(native_eval), str(label_dir), str(pred_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"official_kitti_native_eval_failed exit={proc.returncode}\n{proc.stdout}")
    return proc.stdout.strip()


def _validate_prediction_lines(lines: List[str]) -> None:
    scores = []
    bboxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 16:
            raise RuntimeError(f"Invalid KITTI label_2 line field count: {line}")
        scores.append(float(parts[15]))
        bboxes.append([float(x) for x in parts[4:8]])
    if all(abs(s) < 1e-12 for s in scores):
        raise RuntimeError("All prediction scores are zero.")
    if all(all(abs(v) < 1e-12 for v in box) for box in bboxes):
        raise RuntimeError("All projected 2D bboxes are zero.")


def _atomic_save(obj: Dict[str, object], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, target)


def _load_model_checkpoint(system, path: str, reset_optimizer: bool = False):
    ckpt = torch.load(path, map_location=system.device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_result = system.model.load_state_dict(state, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    refine_missing = [key for key in missing if "head" in key or "score" in key or "fusion" in key]
    print(
        "checkpoint_load "
        f"path={path} epoch={ckpt.get('epoch', 'NA') if isinstance(ckpt, dict) else 'NA'} "
        f"loss_version={ckpt.get('loss_version', 'NA') if isinstance(ckpt, dict) else 'NA'} "
        f"missing={len(missing)} unexpected={len(unexpected)} refine_missing={len(refine_missing)}"
    )
    if isinstance(ckpt, dict) and not reset_optimizer and "optimizer" in ckpt:
        system.optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt if isinstance(ckpt, dict) else {}


def _checkpoint_payload(system, epoch: int, best_loss: float) -> Dict[str, object]:
    return {
        "model": system.model.state_dict(),
        "optimizer": system.optimizer.state_dict(),
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "loss_version": LOSS_VERSION,
        "freeze_report": dict(system.freeze_report),
    }


def _prediction_delta(result: Dict[str, object]) -> Dict[str, float]:
    outputs = result["outputs"]
    raw = outputs.get("raw_prediction", {})
    refined = outputs.get("prediction", {})
    raw_boxes = raw.get("boxes_3d")
    refined_boxes = refined.get("boxes_3d")
    raw_scores = raw.get("scores_3d")
    refined_scores = refined.get("scores_3d")
    deltas = {
        "mean_abs_box_delta": 0.0,
        "max_abs_box_delta": 0.0,
        "mean_abs_score_delta": 0.0,
        "max_abs_score_delta": 0.0,
    }
    if torch.is_tensor(raw_boxes) and torch.is_tensor(refined_boxes) and raw_boxes.numel() and refined_boxes.numel():
        box_delta = (refined_boxes - raw_boxes).abs()
        deltas["mean_abs_box_delta"] = float(box_delta.mean().detach().cpu())
        deltas["max_abs_box_delta"] = float(box_delta.max().detach().cpu())
    if torch.is_tensor(raw_scores) and torch.is_tensor(refined_scores) and raw_scores.numel() and refined_scores.numel():
        score_delta = (refined_scores - raw_scores).abs()
        deltas["mean_abs_score_delta"] = float(score_delta.mean().detach().cpu())
        deltas["max_abs_score_delta"] = float(score_delta.max().detach().cpu())
    return deltas


def run_train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = _resolve_kitti_root(args.kitti_root)
    sample_ids = _read_split(args.train_split_file or args.split_file, root, "train")
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]
    system = build_epnetpp_system(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        box_loss_weight=args.box_loss_weight,
        mc_loss_weight=args.mc_loss_weight,
        point_cls_loss_weight=args.point_cls_loss_weight,
        point_loc_loss_weight=args.point_loc_loss_weight,
        score_refine_loss_weight=args.score_refine_loss_weight,
        score_preserve_loss_weight=args.score_preserve_loss_weight,
        positive_iou_thr=args.positive_iou_thr,
        train_refine_only=args.train_refine_only,
    )
    start_epoch = 0
    best_loss = float("inf")
    if args.resume_ckpt:
        ckpt = _load_model_checkpoint(system, args.resume_ckpt, reset_optimizer=args.reset_optimizer)
        start_epoch = int(ckpt.get("epoch", 0))
        ckpt_loss_version = str(ckpt.get("loss_version", ""))
        if args.reset_best_loss or ckpt_loss_version != LOSS_VERSION:
            best_loss = float("inf")
        else:
            best_loss = float(ckpt.get("best_loss", float("inf")))
    if args.save_best_ckpt and args.save_last_ckpt and Path(args.save_best_ckpt) == Path(args.save_last_ckpt):
        raise ValueError("--save_best_ckpt and --save_last_ckpt must be different paths.")
    print(f"freeze {system.freeze_report}")
    print(f"loss_version {LOSS_VERSION}")
    for epoch in range(start_epoch, args.epochs):
        epoch_ids = list(sample_ids)
        random.Random(args.seed + epoch).shuffle(epoch_ids)
        loss_sum = 0.0
        grad_steps = 0
        skipped = 0
        for sample_idx, sample_id in enumerate(epoch_ids, start=1):
            batch = _build_batch(root, sample_id, "train", max_points=args.max_points)
            losses = train_step(system, batch, grad_clip=args.grad_clip)
            skip = int(float(losses.get("skipped_no_grad", torch.zeros(())).detach().cpu()))
            if skip:
                skipped += 1
            else:
                grad_steps += 1
                loss_sum += float(losses["loss"].detach().cpu())
            loss_text = " ".join(
                f"{key}={float(value):.6f}"
                for key, value in losses.items()
                if torch.is_tensor(value) and value.numel() == 1
            )
            if sample_idx == 1 or sample_idx % args.log_interval == 0 or sample_idx == len(epoch_ids):
                print(
                    f"epoch={epoch + 1} iter={sample_idx}/{len(epoch_ids)} "
                    f"sample={sample_id} skip={skip} grad_steps={grad_steps} {loss_text}"
                )
        mean_loss = loss_sum / max(grad_steps, 1)
        improved = grad_steps > 0 and mean_loss < best_loss
        print(
            f"epoch_summary epoch={epoch + 1} mean_loss={mean_loss:.6f} "
            f"best_loss_before={best_loss:.6f} improved={int(improved)} "
            f"grad_steps={grad_steps} skipped={skipped}"
        )
        payload = _checkpoint_payload(system, epoch + 1, best_loss=min(best_loss, mean_loss))
        if args.save_last_ckpt:
            _atomic_save(payload, args.save_last_ckpt)
        if args.save_ckpt:
            _atomic_save(payload, args.save_ckpt)
        if args.save_best_ckpt and improved:
            best_loss = mean_loss
            payload = _checkpoint_payload(system, epoch + 1, best_loss=best_loss)
            _atomic_save(payload, args.save_best_ckpt)
            print(f"best_checkpoint_saved path={args.save_best_ckpt} best_loss={best_loss:.6f}")


def run_test(args):
    root = _resolve_kitti_root(args.kitti_root)
    sample_ids = _read_split(args.test_split_file or args.split_file, root, "test")
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]
    pred_dir = Path(args.pred_dir)
    data_dir = pred_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir = None
    if args.save_raw_pred_dir:
        raw_data_dir = Path(args.save_raw_pred_dir) / "data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
    system = build_epnetpp_system(
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        box_loss_weight=args.box_loss_weight,
        mc_loss_weight=args.mc_loss_weight,
        point_cls_loss_weight=args.point_cls_loss_weight,
        point_loc_loss_weight=args.point_loc_loss_weight,
        score_refine_loss_weight=args.score_refine_loss_weight,
        score_preserve_loss_weight=args.score_preserve_loss_weight,
        positive_iou_thr=args.positive_iou_thr,
        train_refine_only=args.train_refine_only,
    )
    if args.fusion_ckpt:
        _load_model_checkpoint(system, args.fusion_ckpt, reset_optimizer=True)
    print(f"freeze {system.freeze_report}")
    total_predictions = 0
    empty_samples: List[str] = []
    delta_sum: Dict[str, float] = {
        "mean_abs_box_delta": 0.0,
        "max_abs_box_delta": 0.0,
        "mean_abs_score_delta": 0.0,
        "max_abs_score_delta": 0.0,
    }
    for sample_id in sample_ids:
        batch = _build_batch(root, sample_id, "test", max_points=args.max_points)
        result = test_step(system, batch)
        lines = result["label_2_lines"]
        if lines:
            _validate_prediction_lines(lines)
        else:
            empty_samples.append(sample_id)
        out_path = data_dir / f"{sample_id}.txt"
        out_path.write_text(("\n".join(lines) + "\n") if lines else "")
        total_predictions += len(lines)
        deltas = _prediction_delta(result)
        for key, value in deltas.items():
            delta_sum[key] = max(delta_sum[key], value) if key.startswith("max") else delta_sum[key] + value
        if raw_data_dir is not None:
            raw_outputs = {"prediction": result["outputs"].get("raw_prediction", result["outputs"]["prediction"])}
            raw_lines = system.postprocessor(raw_outputs, batch)
            raw_path = raw_data_dir / f"{sample_id}.txt"
            raw_path.write_text(("\n".join(raw_lines) + "\n") if raw_lines else "")
        if args.debug_refine:
            print(
                f"sample={sample_id} predictions={len(lines)} file={out_path} "
                f"mean_abs_box_delta={deltas['mean_abs_box_delta']:.6f} "
                f"max_abs_box_delta={deltas['max_abs_box_delta']:.6f} "
                f"mean_abs_score_delta={deltas['mean_abs_score_delta']:.6f} "
                f"max_abs_score_delta={deltas['max_abs_score_delta']:.6f}"
            )
        else:
            print(f"sample={sample_id} predictions={len(lines)} file={out_path}")
    if total_predictions <= 0:
        raise RuntimeError("All prediction files are empty; refusing evaluator.")
    denom = max(len(sample_ids), 1)
    print(
        f"test_summary samples={len(sample_ids)} total_predictions={total_predictions} "
        f"empty_prediction_samples={len(empty_samples)} "
        f"mean_abs_box_delta={delta_sum['mean_abs_box_delta'] / denom:.6f} "
        f"max_abs_box_delta={delta_sum['max_abs_box_delta']:.6f} "
        f"mean_abs_score_delta={delta_sum['mean_abs_score_delta'] / denom:.6f} "
        f"max_abs_score_delta={delta_sum['max_abs_score_delta']:.6f}"
    )
    try:
        eval_result = evaluate_predictions(root, pred_dir)
        print(f"evaluator_stdout={eval_result}")
        if args.save_raw_pred_dir:
            raw_eval_result = evaluate_predictions(root, Path(args.save_raw_pred_dir))
            print(f"raw_evaluator_stdout={raw_eval_result}")
    except Exception as exc:
        print(f"evaluator_error={type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="EPNet++ train/test")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--train_split_file", default=None)
    parser.add_argument("--test_split_file", default=None)
    parser.add_argument("--pred_dir", default="outputs/epnetpp_pred")
    parser.add_argument("--save_raw_pred_dir", default=None)
    parser.add_argument("--save_ckpt", default=None)
    parser.add_argument("--save_best_ckpt", default=None)
    parser.add_argument("--save_last_ckpt", default=None)
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument("--fusion_ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--box_loss_weight", type=float, default=0.1)
    parser.add_argument("--point_cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--point_loc_loss_weight", type=float, default=1.0)
    parser.add_argument("--mc_loss_weight", type=float, default=0.2)
    parser.add_argument("--score_refine_loss_weight", type=float, default=0.5)
    parser.add_argument("--score_preserve_loss_weight", type=float, default=0.05)
    parser.add_argument("--positive_iou_thr", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--reset_best_loss", action="store_true")
    parser.add_argument("--debug_refine", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--train_refine_only", action="store_true")
    parser.add_argument("--proposal_pos_iou_thr", type=float, default=0.35)
    parser.add_argument("--proposal_neg_iou_thr", type=float, default=0.15)
    parser.add_argument("--force_match_min_iou", type=float, default=0.15)
    parser.add_argument("--proposal_box_weight", type=float, default=3.0)
    parser.add_argument("--proposal_cls_weight", type=float, default=1.0)
    parser.add_argument("--proposal_loc_weight", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"mmdet={MMDET_MODEL_NAME} config={MMDET_CONFIG} checkpoint={MMDET_CHECKPOINT}")
    print(f"mmdet3d={MMDET3D_MODEL_NAME} config={MMDET3D_CONFIG} checkpoint={MMDET3D_CHECKPOINT}")
    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
