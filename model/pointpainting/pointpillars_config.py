# -*- coding: utf-8 -*-
"""Build reproducible MMDetection3D PointPillars configs for PointPainting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mmengine
from mmengine import Config


def create_annotation_subset(source: str, target: Path, count: int) -> str:
    """Write an MMDetection3D v2 annotation subset for short-loop metrics."""
    annotations = mmengine.load(source)
    if not isinstance(annotations, dict) or "data_list" not in annotations:
        raise ValueError(f"expected v2 annotation dict: {source}")
    subset = dict(annotations)
    subset["data_list"] = annotations["data_list"][: int(count)]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + ".tmp")
    mmengine.dump(subset, temporary, file_format="pkl")
    os.replace(temporary, target)
    return str(target)


def default_pointpillars_config() -> str:
    import mmdet3d

    return str(
        Path(mmdet3d.__file__).resolve().parent
        / ".mim/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
    )


def _configure_dataset(
    dataset: dict,
    kitti_root: str,
    ann_file: str,
    semantic_cache_dir: str,
    painted: bool,
    max_samples: int,
) -> None:
    dataset["data_root"] = str(Path(kitti_root)) + "/"
    dataset["ann_file"] = ann_file
    dataset["data_prefix"] = {"pts": "training/velodyne"}
    dataset["metainfo"] = {"classes": ["Car"]}
    if max_samples > 0:
        dataset["indices"] = int(max_samples)
    pipeline = []
    for step in dataset["pipeline"]:
        if step["type"] == "ObjectSample":
            continue
        if step["type"] == "LoadPointsFromFile":
            step = dict(step)
            step["load_dim"] = 4
            step["use_dim"] = 4
            pipeline.append(step)
            if painted:
                pipeline.append(
                    {"type": "AppendCachedPointSemantics", "cache_dir": semantic_cache_dir}
                )
            continue
        pipeline.append(step)
    dataset["pipeline"] = pipeline


def build_pointpillars_config(
    kitti_root: str,
    train_ann_file: str,
    val_ann_file: str,
    semantic_cache_dir: str,
    work_dir: str,
    epochs: int,
    lr: float,
    painted: bool = True,
    base_config: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: int = 0,
    submission_prefix: Optional[str] = None,
) -> Config:
    config = Config.fromfile(base_config or default_pointpillars_config())
    config.custom_imports = {
        "imports": ["model.pointpainting.mmdet3d_painted"],
        "allow_failed_imports": False,
    }
    config.work_dir = work_dir
    config.model.voxel_encoder.in_channels = 8 if painted else 4

    train_wrapper = config.train_dataloader.dataset
    train_wrapper["times"] = 1
    _configure_dataset(
        train_wrapper["dataset"], kitti_root, train_ann_file,
        semantic_cache_dir, painted, max_samples,
    )
    config.train_dataloader.batch_size = int(batch_size)
    config.train_dataloader.num_workers = int(num_workers)
    config.train_dataloader.persistent_workers = bool(num_workers)

    evaluator_ann_file = val_ann_file
    if max_samples > 0:
        evaluator_ann_file = create_annotation_subset(
            val_ann_file,
            Path(work_dir) / "_meta" / f"val_first_{int(max_samples)}.pkl",
            max_samples,
        )

    _configure_dataset(
        config.val_dataloader.dataset, kitti_root, evaluator_ann_file,
        semantic_cache_dir, painted, max_samples,
    )
    config.val_dataloader.num_workers = min(int(num_workers), 2)
    config.val_dataloader.persistent_workers = bool(num_workers)
    config.test_dataloader = config.val_dataloader

    config.train_cfg.max_epochs = int(epochs)
    config.train_cfg.val_interval = 1
    config.optim_wrapper.optimizer.lr = float(lr)
    config.param_scheduler = {
        "type": "CosineAnnealingLR",
        "T_max": max(int(epochs), 1),
        "eta_min": float(lr) * 0.01,
        "by_epoch": True,
    }
    config.default_hooks.checkpoint.interval = 1
    config.default_hooks.checkpoint.max_keep_ckpts = 5
    config.default_hooks.checkpoint.save_last = True
    config.default_hooks.checkpoint.save_best = (
        "Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict"
    )
    config.default_hooks.checkpoint.rule = "greater"
    config.val_evaluator.ann_file = evaluator_ann_file
    config.val_evaluator.pklfile_prefix = str(Path(work_dir) / "predictions")
    if submission_prefix:
        config.val_evaluator.submission_prefix = submission_prefix
    config.test_evaluator = config.val_evaluator
    # CuBLAS-backed PointPillars kernels are not deterministic on the server's
    # CUDA 11 stack. Keep the seed while following the official fast path.
    config.randomness = {"seed": 0, "deterministic": False}
    return config
