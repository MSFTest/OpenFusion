# -*- coding: utf-8 -*-
"""为 Faraway-Frustum 独立 PV-RCNN 训练准备 KITTI 索引与轻量链接。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def read_split(path: Path) -> List[str]:
    """读取 KITTI sample id，并拒绝重复或非法编号。"""
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"split contains duplicate sample ids: {path}")
    invalid = [sample_id for sample_id in ids if len(sample_id) != 6 or not sample_id.isdigit()]
    if invalid:
        raise ValueError(f"invalid KITTI sample ids in {path}: {invalid[:3]}")
    return ids


def validate_split_ids(train_ids: Sequence[str], val_ids: Sequence[str]) -> None:
    """确保训练集和验证集非空且无交叉，避免 AP 数据泄漏。"""
    if not train_ids or not val_ids:
        raise ValueError("train and val splits must both be non-empty")
    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise ValueError(f"train/val overlap detected: {sorted(overlap)[:3]}")


def _link_directory(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"existing symlink has wrong target: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def _write_ids(path: Path, ids: Iterable[str]) -> None:
    values = list(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{sample_id}\n" for sample_id in values)
    path.write_text(text, encoding="ascii")


def prepare_kitti_layout(
    kitti_root: Path,
    output_root: Path,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> None:
    """创建只含符号链接的数据根目录，不复制图像或点云。"""
    validate_split_ids(train_ids, val_ids)
    for split in ("training", "testing"):
        source_split = kitti_root / split
        if not source_split.is_dir():
            continue
        for name in ("velodyne", "image_2", "label_2", "calib", "planes"):
            source = source_split / name
            if source.is_dir():
                _link_directory(source, output_root / split / name)
        if split == "training":
            _link_directory(source_split / "velodyne", output_root / split / "velodyne_reduced")
    _write_ids(output_root / "ImageSets/train.txt", train_ids)
    _write_ids(output_root / "ImageSets/val.txt", val_ids)
    _write_ids(output_root / "ImageSets/trainval.txt", [*train_ids, *val_ids])
    _write_ids(output_root / "ImageSets/test.txt", [])


def _mmdet3d_tools_root() -> Path:
    spec = importlib.util.find_spec("mmdet3d")
    if spec is None or not spec.origin:
        raise RuntimeError("mmdet3d is required to prepare PV-RCNN data")
    root = Path(spec.origin).resolve().parent / ".mim" / "tools"
    if not root.is_dir():
        raise RuntimeError(f"mmdet3d conversion tools not found: {root}")
    return root


def is_v2_info_file(path: Path) -> bool:
    """判断 info 是否已经是 mmdet3d v2 字典格式。"""
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        value = pickle.load(stream)
    return isinstance(value, dict) and "metainfo" in value and "data_list" in value


def create_info_files(output_root: Path, info_prefix: str, with_gt_database: bool) -> None:
    """生成 mmdet3d v2 KITTI info；按需生成目标采样数据库。"""
    tools_root = _mmdet3d_tools_root()
    sys.path.insert(0, str(tools_root))
    try:
        from mmdet3d.utils import register_all_modules
        from dataset_converters import kitti_converter
        from dataset_converters.create_gt_database import create_groundtruth_database
        from dataset_converters.update_infos_to_v2 import update_pkl_infos

        register_all_modules()
        root = str(output_root.resolve())
        required_infos = [
            output_root / f"{info_prefix}_infos_train.pkl",
            output_root / f"{info_prefix}_infos_val.pkl",
        ]
        if not all(is_v2_info_file(path) for path in required_infos):
            kitti_converter.create_kitti_info_file(root, info_prefix, save_path=root, with_plane=False)
        for split in ("train", "val"):
            info_path = output_root / f"{info_prefix}_infos_{split}.pkl"
            if not is_v2_info_file(info_path):
                update_pkl_infos("kitti", out_dir=root, pkl_path=str(info_path))
        if with_gt_database:
            create_groundtruth_database(
                "KittiDataset",
                root,
                info_prefix,
                f"{info_prefix}_infos_train.pkl",
                relative_path=True,
            )
    finally:
        if sys.path and sys.path[0] == str(tools_root):
            sys.path.pop(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--train_split_file", required=True)
    parser.add_argument("--val_split_file", required=True)
    parser.add_argument("--info_prefix", default="kitti")
    parser.add_argument("--with_gt_database", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ids = read_split(Path(args.train_split_file))
    val_ids = read_split(Path(args.val_split_file))
    output_root = Path(args.output_root)
    prepare_kitti_layout(Path(args.kitti_root), output_root, train_ids, val_ids)
    create_info_files(output_root, args.info_prefix, args.with_gt_database)
    print(
        f"prepared_pvrcnn_data={output_root.resolve()} "
        f"train={len(train_ids)} val={len(val_ids)} gt_database={int(args.with_gt_database)}"
    )


if __name__ == "__main__":
    main()
