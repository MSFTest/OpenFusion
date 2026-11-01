# -*- coding: utf-8 -*-
"""Generate compact per-point semantic caches for KITTI PointPainting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.pointpainting.kitti_painting import encode_point_semantics


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def cache_is_valid(path: Path, expected_points: int) -> bool:
    try:
        values = np.load(str(path), mmap_mode="r", allow_pickle=False)
        return (
            values.shape == (int(expected_points), 4)
            and values.dtype == np.uint8
        )
    except (OSError, ValueError, EOFError):
        return False


def _read_calibration(path: Path) -> Dict[str, np.ndarray]:
    values: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            values[key] = np.asarray([float(item) for item in raw.split()], dtype=np.float32)
    return {
        "P2": values["P2"].reshape(3, 4),
        "R0_rect": values["R0_rect"].reshape(3, 3),
        "Tr_velo_to_cam": values.get("Tr_velo_to_cam", values.get("Tr_velo2cam")).reshape(3, 4),
    }


def _prediction_arrays(prediction, score_threshold: float):
    instances = prediction.pred_instances
    scores = instances.scores.detach().cpu().numpy()
    keep = scores >= float(score_threshold)
    labels = instances.labels.detach().cpu().numpy()[keep]
    scores = scores[keep]
    masks = instances.masks
    if hasattr(masks, "to_tensor"):
        masks = masks.to_tensor(dtype=np.bool_, device="cpu")
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    return np.asarray(masks, dtype=bool)[keep], labels, scores


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_cfg", required=True)
    parser.add_argument("--image_ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules

    register_all_modules()
    model = init_detector(args.image_cfg, args.image_ckpt, device=args.device)
    root = Path(args.kitti_root)
    output_dir = Path(args.output_dir)
    sample_ids = [
        line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]

    generated = skipped = 0
    for index, sample_id in enumerate(sample_ids, start=1):
        point_path = root / "training" / "velodyne" / f"{sample_id}.bin"
        image_path = root / "training" / "image_2" / f"{sample_id}.png"
        calib_path = root / "training" / "calib" / f"{sample_id}.txt"
        points = np.fromfile(str(point_path), dtype=np.float32).reshape(-1, 4)
        target = output_dir / f"{sample_id}.npy"
        if not args.overwrite and cache_is_valid(target, points.shape[0]):
            skipped += 1
            continue
        prediction = inference_detector(model, str(image_path))
        masks, labels, confidences = _prediction_arrays(prediction, args.score_threshold)
        calibration = _read_calibration(calib_path)
        with Image.open(image_path) as image:
            image_shape = (image.height, image.width)
        semantics = encode_point_semantics(
            points,
            calibration["Tr_velo_to_cam"],
            calibration["R0_rect"],
            calibration["P2"],
            masks,
            labels,
            confidences,
            image_shape,
        )
        atomic_save_npy(target, semantics)
        generated += 1
        if index == 1 or index % 20 == 0:
            foreground = float((semantics[:, 1:].max(axis=1) > 0).mean())
            coverage = [int((semantics[:, channel] > 0).sum()) for channel in range(1, 4)]
            print(
                f"[PointPainting][cache] {index}/{len(sample_ids)} id={sample_id} "
                f"masks={len(masks)} foreground_ratio={foreground:.6f} coverage={coverage}",
                flush=True,
            )
    print(f"[PointPainting][cache] generated={generated} skipped={skipped}")


if __name__ == "__main__":
    main()
