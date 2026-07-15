# -*- coding: utf-8 -*-
"""EPNet train/test entrypoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.epnet.initialization.build_epnet_system import (
    MMDET3D_CHECKPOINT,
    MMDET3D_CONFIG,
    MMDET3D_MODEL_NAME,
    MMDET_CHECKPOINT,
    MMDET_CONFIG,
    MMDET_MODEL_NAME,
    build_epnet_system,
    test_step,
    train_step,
)

LOSS_VERSION = "epnet_proposal_refine_v1"


def _atomic_torch_save(obj, path: str) -> None:
    """原子保存 checkpoint，避免中断时把已有权重覆盖成半截 zip。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target)


def _resolve_kitti_root(root: str) -> Path:
    base = Path(root)
    candidates = [base, base / "object"]
    for cand in candidates:
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
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _load_points(path: Path, max_points: int = 4096) -> torch.Tensor:
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]
    return torch.from_numpy(points).float().unsqueeze(0)


def _projection_mats(calib: Dict[str, torch.Tensor]):
    r0 = calib["R0_rect"]
    tr = calib["Tr_velo2cam"]
    return r0 @ tr[:, :3], r0 @ tr[:, 3], calib["P2"][:, :3]


def _read_label_boxes(path: Path, calib: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not path.exists():
        return torch.zeros((0, 7), dtype=torch.float32)
    r0_4 = torch.eye(4, dtype=torch.float32)
    tr_4 = torch.eye(4, dtype=torch.float32)
    r0_4[:3, :3] = calib["R0_rect"].to(dtype=torch.float32)
    tr_4[:3, :4] = calib["Tr_velo2cam"].to(dtype=torch.float32)
    rect_to_lidar = torch.inverse(r0_4 @ tr_4)
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 15 or parts[0] == "DontCare":
            continue
        h, w, l = [float(x) for x in parts[8:11]]
        x, y, z = [float(x) for x in parts[11:14]]
        ry = float(parts[14])
        center_cam = torch.tensor([x, y, z, 1.0], dtype=torch.float32)
        center_lidar = (rect_to_lidar @ center_cam)[:3]
        boxes.append([
            float(center_lidar[0]),
            float(center_lidar[1]),
            float(center_lidar[2]),
            l,
            w,
            h,
            -ry - np.pi / 2.0,
        ])
    if not boxes:
        return torch.zeros((0, 7), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _build_batch(root: Path, sample_id: str, mode: str) -> Dict[str, object]:
    image_path = _find_file(root, sample_id, ["training/image_2/{id}.png", "testing/image_2/{id}.png"])
    velo_path = _find_file(root, sample_id, ["training/velodyne/{id}.bin", "testing/velodyne/{id}.bin"])
    calib_path = _find_file(root, sample_id, ["training/calib/{id}.txt", "testing/calib/{id}.txt"])
    calib = _read_calib(calib_path)
    r, t, k = _projection_mats(calib)
    batch: Dict[str, object] = {
        "sample_id": sample_id,
        "pcd_path": str(velo_path),
        "images": _load_image(image_path),
        "points": _load_points(velo_path),
        "calib": calib,
        "lidar_to_cam_r": r.unsqueeze(0),
        "lidar_to_cam_t": t.unsqueeze(0),
        "cam_intrinsic": k.unsqueeze(0),
    }
    label_candidates = [root / "training" / "label_2" / f"{sample_id}.txt"]
    if mode == "train":
        label_path = label_candidates[0]
        if not label_path.exists():
            raise FileNotFoundError(f"Training label_2 not found: {label_path}")
        batch["gt_boxes_3d"] = _read_label_boxes(label_path, calib)
    return batch


def _parse_label_file(path: Path):
    names, truncated, occluded, alpha, bbox, dims, loc, rots, scores = [], [], [], [], [], [], [], [], []
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 15:
            continue
        names.append(parts[0])
        truncated.append(float(parts[1]))
        occluded.append(int(float(parts[2])))
        alpha.append(float(parts[3]))
        bbox.append([float(x) for x in parts[4:8]])
        dims.append([float(x) for x in parts[8:11]])
        loc.append([float(x) for x in parts[11:14]])
        rots.append(float(parts[14]))
        scores.append(float(parts[15]) if len(parts) > 15 else 1.0)
    return {
        "name": np.array(names),
        "truncated": np.array(truncated, dtype=np.float32),
        "occluded": np.array(occluded, dtype=np.int64),
        "alpha": np.array(alpha, dtype=np.float32),
        "bbox": np.array(bbox, dtype=np.float32).reshape(-1, 4),
        "dimensions": np.array(dims, dtype=np.float32).reshape(-1, 3),
        "location": np.array(loc, dtype=np.float32).reshape(-1, 3),
        "rotation_y": np.array(rots, dtype=np.float32),
        "score": np.array(scores, dtype=np.float32),
    }


def evaluate_predictions(root: Path, pred_dir: Path, sample_ids: List[str]):
    """调用仓库中的 KITTI native 官方评测程序。"""
    label_dir = root / "training" / "label_2"
    if not label_dir.exists():
        raise RuntimeError(f"KITTI label_2 directory not found: {label_dir}")
    native_eval = REPO_ROOT / "kitti_native_evaluation" / "build" / "evaluate_object_3d_offline"
    if not native_eval.exists():
        raise RuntimeError(f"KITTI native evaluator not found: {native_eval}")
    proc = subprocess.run(
        [str(native_eval), str(label_dir), str(pred_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"official_kitti_native_eval_failed exit={proc.returncode}\n{proc.stdout}")
    stats = {}
    for path in sorted(pred_dir.glob("stats_*.txt")):
        rows = []
        for line in path.read_text().splitlines():
            vals = [float(x) for x in line.split()]
            if vals:
                rows.append(sum(vals) / len(vals) * 100.0)
        stats[path.name] = rows
    return {"stdout": proc.stdout.strip(), "stats": stats}


def run_train(args):
    root = _resolve_kitti_root(args.kitti_root)
    sample_ids = _read_split(args.train_split_file or args.split_file, root, "train")
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]
    system = build_epnet_system(
        device=args.device,
        lr=args.lr,
        proposal_pos_iou_thr=args.proposal_pos_iou_thr,
        proposal_neg_iou_thr=args.proposal_neg_iou_thr,
        proposal_box_weight=args.proposal_box_weight,
        proposal_cls_weight=args.proposal_cls_weight,
        proposal_loc_weight=args.proposal_loc_weight,
        consistency_weight=args.consistency_weight,
        force_match_min_iou=args.force_match_min_iou,
    )
    if args.train_refine_only:
        for name, param in system.model.named_parameters():
            param.requires_grad_("detection_refine" in name)
        trainable_params = [p for p in system.model.parameters() if p.requires_grad]
        system.optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
        print(f"train_refine_only=True optimizer_param_count={sum(p.numel() for p in trainable_params)}")
    start_epoch = 0
    resume_sample_idx = 0
    best_loss = float("inf")
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=system.device)
        missing, unexpected = system.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"resume_missing_keys={len(missing)}")
        if unexpected:
            print(f"resume_unexpected_keys={len(unexpected)}")
        if args.reset_optimizer:
            print("resume_optimizer_skipped=reset_optimizer")
        else:
            try:
                system.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                print(f"resume_optimizer_skipped={exc}")
        start_epoch = int(ckpt.get("epoch", 0))
        resume_sample_idx = int(ckpt.get("sample_idx", 0))
        ckpt_loss_version = ckpt.get("loss_version")
        if args.reset_best_loss:
            print("reset_best_loss requested")
            best_loss = float("inf")
        elif ckpt_loss_version == LOSS_VERSION:
            best_loss = float(ckpt.get("best_loss", best_loss))
        else:
            print(f"reset_best_loss old_loss_version={ckpt_loss_version} current_loss_version={LOSS_VERSION}")
            best_loss = float("inf")
        if resume_sample_idx > 0:
            print(f"resume_mid_epoch epoch={start_epoch + 1} next_sample_idx={resume_sample_idx + 1}")
    print(f"freeze {system.freeze_report}")

    def _save_train_state(path: str, epoch_value: int, sample_idx_value: int, best_loss_value: float) -> None:
        if not path:
            return
        _atomic_torch_save(
            {
                "model": system.model.state_dict(),
                "optimizer": system.optimizer.state_dict(),
                "epoch": epoch_value,
                "sample_idx": sample_idx_value,
                "best_loss": best_loss_value,
                "loss_version": LOSS_VERSION,
            },
            path,
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        start_sample_idx = resume_sample_idx if epoch == start_epoch else 0
        for sample_idx, sample_id in enumerate(sample_ids, start=1):
            if sample_idx <= start_sample_idx:
                continue
            try:
                batch = _build_batch(root, sample_id, "train")
                losses = train_step(system, batch)
            except Exception as exc:
                if args.save_last_ckpt:
                    _save_train_state(args.save_last_ckpt, epoch, sample_idx - 1, best_loss)
                    print(f"saved_last_before_error={args.save_last_ckpt} epoch={epoch + 1} sample_idx={sample_idx - 1}")
                print(f"train_error epoch={epoch + 1} sample_idx={sample_idx}/{len(sample_ids)} sample={sample_id} error={type(exc).__name__}: {exc}")
                raise
            loss_value = float(losses["loss"])
            skipped_no_grad = float(losses.get("skipped_no_grad", 0.0)) > 0.5
            if not skipped_no_grad:
                epoch_losses.append(loss_value)
            if sample_idx % 10 == 0 or sample_idx == len(sample_ids):
                detail = ""
                if args.debug_refine:
                    detail = (
                        f" skip={int(skipped_no_grad)}"
                        f" loss_pcls={float(losses.get('loss_proposal_cls', 0.0)):.6f}"
                        f" loss_pbox={float(losses.get('loss_proposal_box', 0.0)):.6f}"
                        f" loss_ploc={float(losses.get('loss_proposal_loc', 0.0)):.6f}"
                        f" pos={float(losses.get('proposal_pos_num', 0.0)):.0f}"
                        f" neg={float(losses.get('proposal_neg_num', 0.0)):.0f}"
                        f" ign={float(losses.get('proposal_ignore_num', 0.0)):.0f}"
                        f" max_iou={float(losses.get('proposal_max_iou', 0.0)):.4f}"
                    )
                print(f"epoch={epoch + 1} sample_idx={sample_idx}/{len(sample_ids)} sample={sample_id} loss={loss_value:.6f}{detail}")
        epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        print(
            f"epoch={epoch + 1} mean_loss={epoch_loss:.6f} "
            f"best_loss={best_loss:.6f} grad_steps={len(epoch_losses)}"
        )
        state = {
            "model": system.model.state_dict(),
            "optimizer": system.optimizer.state_dict(),
            "epoch": epoch + 1,
            "sample_idx": 0,
            "best_loss": min(best_loss, epoch_loss),
            "loss_version": LOSS_VERSION,
        }
        if args.save_last_ckpt:
            _atomic_torch_save(state, args.save_last_ckpt)
            print(f"saved_last={args.save_last_ckpt}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state["best_loss"] = best_loss
            if args.save_best_ckpt:
                _atomic_torch_save(state, args.save_best_ckpt)
                print(f"saved_best={args.save_best_ckpt} best_loss={best_loss:.6f}")
        resume_sample_idx = 0
    if args.save_ckpt:
        _atomic_torch_save(
            {
                "model": system.model.state_dict(),
                "optimizer": system.optimizer.state_dict(),
                "epoch": args.epochs,
                "best_loss": best_loss,
                "loss_version": LOSS_VERSION,
            },
            args.save_ckpt,
        )


def run_test(args):
    root = _resolve_kitti_root(args.kitti_root)
    sample_ids = _read_split(args.test_split_file or args.split_file, root, "test")
    if args.max_samples:
        sample_ids = sample_ids[: args.max_samples]
    pred_dir = Path(args.pred_dir)
    data_dir = pred_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    system = build_epnet_system(
        device=args.device,
        lr=args.lr,
        proposal_pos_iou_thr=args.proposal_pos_iou_thr,
        proposal_neg_iou_thr=args.proposal_neg_iou_thr,
        proposal_box_weight=args.proposal_box_weight,
        proposal_cls_weight=args.proposal_cls_weight,
        proposal_loc_weight=args.proposal_loc_weight,
        consistency_weight=args.consistency_weight,
        force_match_min_iou=args.force_match_min_iou,
    )
    if args.fusion_ckpt:
        ckpt = torch.load(args.fusion_ckpt, map_location=system.device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = system.model.load_state_dict(state_dict, strict=False)
        print(
            f"loaded_fusion_ckpt={args.fusion_ckpt} "
            f"epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None} "
            f"loss_version={ckpt.get('loss_version') if isinstance(ckpt, dict) else None}"
        )
        if missing:
            refine_missing = [key for key in missing if "detection_refine" in key or "proposal" in key]
            print(f"fusion_ckpt_missing_keys={len(missing)} proposal_refine_missing={len(refine_missing)}")
            if refine_missing:
                print("fusion_ckpt_missing_refine_sample=" + ",".join(refine_missing[:8]))
        if unexpected:
            print(f"fusion_ckpt_unexpected_keys={len(unexpected)}")
    print(f"freeze {system.freeze_report}")
    raw_data_dir = None
    if args.save_raw_pred_dir:
        raw_data_dir = Path(args.save_raw_pred_dir) / "data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    debug_box_deltas = []
    debug_score_deltas = []
    total_predictions = 0
    empty_prediction_samples = []
    for sample_id in sample_ids:
        batch = _build_batch(root, sample_id, "test")
        result = test_step(system, batch)
        out_path = data_dir / f"{sample_id}.txt"
        lines = result["label_2_lines"]
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        if not lines:
            empty_prediction_samples.append(sample_id)
        total_predictions += len(lines)
        if raw_data_dir is not None:
            raw_out_path = raw_data_dir / f"{sample_id}.txt"
            raw_lines = result["raw_label_2_lines"]
            raw_out_path.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""))
        if args.debug_refine:
            outputs = result["outputs"]
            raw = outputs["detector_prediction"]
            refined = outputs["prediction"]
            if raw["boxes_3d"].numel() and refined["boxes_3d"].numel():
                n = min(raw["boxes_3d"].shape[0], refined["boxes_3d"].shape[0])
                debug_box_deltas.append((refined["boxes_3d"][:n] - raw["boxes_3d"][:n]).abs().detach().cpu())
                debug_score_deltas.append((refined["scores_3d"][:n] - raw["scores_3d"][:n]).abs().detach().cpu())
        written.append(sample_id)
        print(f"sample={sample_id} predictions={len(lines)} file={out_path}")
    if total_predictions <= 0:
        raise RuntimeError("Detector produced no KITTI lines for the full split; refusing all-empty output.")
    if empty_prediction_samples:
        print(
            f"empty_prediction_samples={len(empty_prediction_samples)}/{len(sample_ids)} "
            f"first_empty={','.join(empty_prediction_samples[:10])}"
        )
    if args.debug_refine and debug_box_deltas:
        box_delta = torch.cat(debug_box_deltas, dim=0)
        score_delta = torch.cat(debug_score_deltas, dim=0)
        print(
            "refine_debug "
            f"num_predictions={box_delta.shape[0]} "
            f"mean_abs_box_delta={float(box_delta.mean()):.6f} "
            f"max_abs_box_delta={float(box_delta.max()):.6f} "
            f"mean_abs_score_delta={float(score_delta.mean()):.6f} "
            f"max_abs_score_delta={float(score_delta.max()):.6f}"
        )
    try:
        eval_result = evaluate_predictions(root, pred_dir, written)
        print(f"evaluator_stdout={eval_result['stdout']}")
        print(f"evaluator_stats={eval_result['stats']}")
    except Exception as exc:
        print(f"evaluator_error={type(exc).__name__}: {exc}")
    if args.save_raw_pred_dir:
        try:
            raw_eval = evaluate_predictions(root, Path(args.save_raw_pred_dir), written)
            print(f"raw_evaluator_stdout={raw_eval['stdout']}")
            print(f"raw_evaluator_stats={raw_eval['stats']}")
        except Exception as exc:
            print(f"raw_evaluator_error={type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="EPNet train/test")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--train_split_file", default=None)
    parser.add_argument("--test_split_file", default=None)
    parser.add_argument("--pred_dir", default="outputs/epnet_pred")
    parser.add_argument("--save_raw_pred_dir", default=None)
    parser.add_argument("--save_ckpt", default=None)
    parser.add_argument("--save_best_ckpt", default="checkpoints/epnet_fusion_best.pth")
    parser.add_argument("--save_last_ckpt", default="checkpoints/epnet_fusion_last.pth")
    parser.add_argument("--resume_ckpt", default=None)
    parser.add_argument("--fusion_ckpt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--debug_refine", action="store_true")
    parser.add_argument("--train_refine_only", action="store_true")
    parser.add_argument("--proposal_pos_iou_thr", type=float, default=0.55)
    parser.add_argument("--proposal_neg_iou_thr", type=float, default=0.25)
    parser.add_argument("--proposal_box_weight", type=float, default=1.0)
    parser.add_argument("--proposal_cls_weight", type=float, default=1.0)
    parser.add_argument("--proposal_loc_weight", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
    parser.add_argument("--force_match_min_iou", type=float, default=0.15)
    parser.add_argument("--reset_best_loss", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
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
