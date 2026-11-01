# -*- coding: utf-8 -*-
"""Faraway-Frustum 论文远距离 KITTI AP11 评测。"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import numpy as np


def _get_thresholds_ap11(scores: np.ndarray, num_gt: int) -> List[float]:
    """复现原项目 C++ evaluator 的 11 点置信度阈值采样。"""
    if num_gt <= 0:
        return []
    ordered = sorted((float(score) for score in scores), reverse=True)
    current_recall = 0.0
    thresholds: List[float] = []
    for index, score in enumerate(ordered):
        left_recall = float(index + 1) / float(num_gt)
        right_recall = (
            float(index + 2) / float(num_gt)
            if index < len(ordered) - 1
            else left_recall
        )
        if (
            right_recall - current_recall < current_recall - left_recall
            and index < len(ordered) - 1
        ):
            continue
        thresholds.append(score)
        current_recall += 0.1
    return thresholds


def _evaluate_metric_ap11(
    gt_annos: Sequence[Dict[str, np.ndarray]],
    pred_annos: Sequence[Dict[str, np.ndarray]],
    current_class: int,
    metric: int,
    min_iou: float,
) -> float:
    """按原项目 N_SAMPLE_PTS=11 的 KITTI 匹配规则计算单项 AP。"""
    from mmdet3d.evaluation.functional.kitti_utils.eval import (
        _prepare_data,
        calculate_iou_partly,
        compute_statistics_jit,
        fused_compute_statistics,
        get_split_parts,
    )

    num_examples = len(gt_annos)
    num_parts = min(200, num_examples)
    split_parts = get_split_parts(num_examples, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = calculate_iou_partly(
        pred_annos, gt_annos, metric, num_parts
    )
    prepared = _prepare_data(gt_annos, pred_annos, current_class, 0)
    (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        total_dc_num,
        total_num_valid_gt,
    ) = prepared
    threshold_scores: List[float] = []
    for index in range(num_examples):
        statistics = compute_statistics_jit(
            overlaps[index],
            gt_datas_list[index],
            dt_datas_list[index],
            ignored_gts[index],
            ignored_dets[index],
            dontcares[index],
            metric,
            min_overlap=float(min_iou),
            thresh=0.0,
            compute_fp=False,
        )
        threshold_scores.extend(statistics[4].tolist())
    thresholds = np.asarray(
        _get_thresholds_ap11(np.asarray(threshold_scores), total_num_valid_gt),
        dtype=np.float64,
    )
    precision = np.zeros(11, dtype=np.float64)
    if thresholds.size == 0:
        return 0.0
    pr = np.zeros((thresholds.size, 4), dtype=np.float64)
    offset = 0
    for part_index, num_part in enumerate(split_parts):
        end = offset + num_part
        fused_compute_statistics(
            parted_overlaps[part_index],
            pr,
            total_gt_num[offset:end],
            total_dt_num[offset:end],
            total_dc_num[offset:end],
            np.concatenate(gt_datas_list[offset:end], axis=0),
            np.concatenate(dt_datas_list[offset:end], axis=0),
            np.concatenate(dontcares[offset:end], axis=0),
            np.concatenate(ignored_gts[offset:end], axis=0),
            np.concatenate(ignored_dets[offset:end], axis=0),
            metric,
            min_overlap=float(min_iou),
            thresholds=thresholds,
            compute_aos=False,
        )
        offset = end
    valid = min(thresholds.size, precision.size)
    denominator = pr[:valid, 0] + pr[:valid, 1]
    precision[:valid] = np.divide(
        pr[:valid, 0],
        denominator,
        out=np.zeros(valid, dtype=np.float64),
        where=denominator > 0,
    )
    for index in range(valid):
        precision[index] = np.max(precision[index:])
    return float(precision.mean() * 100.0)


def read_kitti_anno(path: str) -> Dict[str, np.ndarray]:
    """读取 KITTI label/prediction 文件。

    Args:
        path: KITTI 文本文件路径。每行为相机坐标系目标，尺寸顺序为 h/w/l。

    Returns:
        标注字典；``bbox`` shape 为 ``(N, 4)``，``dimensions`` 和
        ``location`` shape 均为 ``(N, 3)``。KITTI 文件的 h/w/l 会转换成
        mmdet3d evaluator 所需的 l/h/w。
    """
    fields: Dict[str, List] = {
        "name": [],
        "truncated": [],
        "occluded": [],
        "alpha": [],
        "bbox": [],
        "dimensions": [],
        "location": [],
        "rotation_y": [],
        "score": [],
    }
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                fields["name"].append(parts[0])
                fields["truncated"].append(float(parts[1]))
                fields["occluded"].append(int(float(parts[2])))
                fields["alpha"].append(float(parts[3]))
                fields["bbox"].append([float(value) for value in parts[4:8]])
                height, width, length = (float(value) for value in parts[8:11])
                fields["dimensions"].append([length, height, width])
                fields["location"].append([float(value) for value in parts[11:14]])
                fields["rotation_y"].append(float(parts[14]))
                fields["score"].append(float(parts[15]) if len(parts) > 15 else 1.0)
    return {
        "name": np.asarray(fields["name"]),
        "truncated": np.asarray(fields["truncated"], dtype=np.float32),
        "occluded": np.asarray(fields["occluded"], dtype=np.int64),
        "alpha": np.asarray(fields["alpha"], dtype=np.float32),
        "bbox": np.asarray(fields["bbox"], dtype=np.float32).reshape(-1, 4),
        "dimensions": np.asarray(fields["dimensions"], dtype=np.float32).reshape(-1, 3),
        "location": np.asarray(fields["location"], dtype=np.float32).reshape(-1, 3),
        "rotation_y": np.asarray(fields["rotation_y"], dtype=np.float32),
        "score": np.asarray(fields["score"], dtype=np.float32),
    }


def filter_faraway_anno(
    anno: Dict[str, np.ndarray],
    class_name: str,
    min_depth: float,
) -> Dict[str, np.ndarray]:
    """筛选论文定义的远距离目标并统一专用 benchmark 难度条件。

    Args:
        anno: KITTI 相机坐标标注字典，``location`` shape 为 ``(N, 3)``。
        class_name: 评测类别，例如 ``Car``。
        min_depth: 相机 rect 坐标系 z 方向最小距离，单位米。

    Returns:
        过滤后的标注字典。2D 框高度被设为 50 像素，使 mmdet3d 的
        三档标准难度入口等价于论文远距离 benchmark 的统一难度设置。
    """
    names = np.asarray(anno["name"])
    locations = np.asarray(anno["location"], dtype=np.float32).reshape(-1, 3)
    keep = (names == class_name) & (locations[:, 2] >= float(min_depth))
    count = names.shape[0]
    filtered: Dict[str, np.ndarray] = {}
    for key, value in anno.items():
        array = np.asarray(value)
        filtered[key] = array[keep].copy() if array.shape[:1] == (count,) else array.copy()

    bbox = np.asarray(filtered["bbox"], dtype=np.float32).reshape(-1, 4).copy()
    if bbox.shape[0] > 0:
        bbox[:, 3] = bbox[:, 1] + 50.0
    filtered["bbox"] = bbox
    truncated = np.asarray(filtered["truncated"], dtype=np.float32)
    filtered["truncated"] = np.where(truncated <= 0.5, 0.0, 1.0).astype(np.float32)
    occluded = np.asarray(filtered["occluded"], dtype=np.int64)
    filtered["occluded"] = np.where(occluded <= 2, 0, 3).astype(np.int64)
    return filtered


def evaluate_faraway_kitti(
    gt_annos: Sequence[Dict[str, np.ndarray]],
    pred_annos: Sequence[Dict[str, np.ndarray]],
    class_name: str = "Car",
    min_depth: float = 75.0,
    min_iou: float = 0.1,
) -> Dict[str, float]:
    """计算论文远距离子集的 BEV/3D AP11。

    Args:
        gt_annos: KITTI GT 字典序列，框位于相机 rect 坐标系。
        pred_annos: 同序预测字典序列，shape 与 GT 字段规范一致。
        class_name: 目标类别；论文 Car 指标使用 ``Car``。
        min_depth: 相机 z 方向阈值；论文 Car 指标使用 75 米。
        min_iou: BEV/3D IoU 阈值；论文远距离指标使用 0.1。

    Returns:
        包含 ``bev_ap11``、``3d_ap11`` 和有效帧/GT 数量的字典。
    """
    if len(gt_annos) != len(pred_annos):
        raise ValueError("gt_annos and pred_annos must have the same length")
    selected_gt: List[Dict[str, np.ndarray]] = []
    selected_pred: List[Dict[str, np.ndarray]] = []
    for gt_anno, pred_anno in zip(gt_annos, pred_annos):
        far_gt = filter_faraway_anno(gt_anno, class_name, min_depth)
        if far_gt["name"].shape[0] == 0:
            continue
        selected_gt.append(far_gt)
        selected_pred.append(filter_faraway_anno(pred_anno, class_name, min_depth))
    if not selected_gt:
        raise RuntimeError(f"No {class_name} ground truth found at z >= {min_depth:g} m")

    class_names = ["Car", "Pedestrian", "Cyclist"]
    try:
        current_class = class_names.index(class_name)
    except ValueError as exc:
        raise ValueError(f"Unsupported KITTI class: {class_name}") from exc
    bev_ap11 = _evaluate_metric_ap11(
        selected_gt, selected_pred, current_class, metric=1, min_iou=min_iou
    )
    d3_ap11 = _evaluate_metric_ap11(
        selected_gt, selected_pred, current_class, metric=2, min_iou=min_iou
    )
    return {
        "bev_ap11": bev_ap11,
        "3d_ap11": d3_ap11,
        "frames": float(len(selected_gt)),
        "gt_count": float(sum(anno["name"].shape[0] for anno in selected_gt)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--class_name", default="Car")
    parser.add_argument("--min_depth", type=float, default=75.0)
    parser.add_argument("--min_iou", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.split_file, "r", encoding="utf-8") as handle:
        sample_ids = [line.strip() for line in handle if line.strip()]
    pred_data_dir = os.path.join(args.pred_dir, "data")
    if not os.path.isdir(pred_data_dir):
        pred_data_dir = args.pred_dir
    gt_annos = [
        read_kitti_anno(os.path.join(args.kitti_root, "training", "label_2", f"{sample_id}.txt"))
        for sample_id in sample_ids
    ]
    pred_annos = [
        read_kitti_anno(os.path.join(pred_data_dir, f"{sample_id}.txt"))
        for sample_id in sample_ids
    ]
    result = evaluate_faraway_kitti(
        gt_annos,
        pred_annos,
        class_name=args.class_name,
        min_depth=args.min_depth,
        min_iou=args.min_iou,
    )
    print(
        f"faraway_{args.class_name.lower()}_z{args.min_depth:g}_iou{args.min_iou:g} "
        f"BEV_AP11={result['bev_ap11']:.6f} 3D_AP11={result['3d_ap11']:.6f} "
        f"frames={int(result['frames'])} gt={int(result['gt_count'])}"
    )


if __name__ == "__main__":
    main()
