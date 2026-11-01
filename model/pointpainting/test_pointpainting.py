# -*- coding: utf-8 -*-
"""Native MMDetection3D train/test entry point for Painted PointPillars."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.pointpainting.pointpillars_config import build_pointpillars_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--kitti_root", required=True)
    parser.add_argument("--split_file", default="")
    parser.add_argument("--train_split_file", default="")
    parser.add_argument("--test_split_file", default="")
    parser.add_argument("--train_ann_file", default="")
    parser.add_argument("--test_ann_file", default="")
    parser.add_argument("--semantic_cache_dir", default="")
    parser.add_argument("--pred_dir", default="")
    parser.add_argument("--save_raw_pred_dir", default="")
    parser.add_argument("--base_config", default="")
    parser.add_argument("--image_cfg", default="")
    parser.add_argument("--image_ckpt", default="")
    parser.add_argument("--point_cloud_cfg", default="")
    parser.add_argument("--point_cloud_ckpt", default="")
    parser.add_argument("--save_ckpt", default="")
    parser.add_argument("--save_best_ckpt", default="")
    parser.add_argument("--save_last_ckpt", default="")
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--fusion_ckpt", default="")
    parser.add_argument("--raw_ckpt", default="")
    parser.add_argument(
        "--raw_only", action="store_true", help="Evaluate only the unpainted baseline."
    )
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--debug_refine", action="store_true")
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="Load checkpoint weights only and restart epoch/scheduler/best-metric state.",
    )
    return parser


def _require_file(value: str, argument: str) -> str:
    if not value:
        raise ValueError(f"{argument} is required")
    if not Path(value).is_file():
        raise FileNotFoundError(f"{argument} not found: {value}")
    return value


def validate_split_annotation(split_file: str, ann_file: str, argument: str) -> None:
    """Fail fast if a user-supplied split and v2 annotation file diverge."""
    if not split_file:
        return
    import mmengine

    split_path = Path(_require_file(split_file, argument))
    expected = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotations = mmengine.load(ann_file)
    if not isinstance(annotations, dict) or "data_list" not in annotations:
        raise ValueError(f"expected v2 annotation dict: {ann_file}")
    actual = [f"{int(item['sample_idx']):06d}" for item in annotations["data_list"]]
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
            min(len(actual), len(expected)),
        )
        raise ValueError(
            f"{argument} does not match {ann_file}: split={len(expected)} ann={len(actual)} "
            f"first_mismatch={mismatch}"
        )


def _build_config(args, painted: bool, work_dir: Path, submission_prefix: str = ""):
    train_ann = _require_file(
        args.train_ann_file or args.test_ann_file, "--train_ann_file/--test_ann_file"
    )
    val_ann = _require_file(args.test_ann_file, "--test_ann_file")
    validate_split_annotation(args.train_split_file, train_ann, "--train_split_file")
    validate_split_annotation(
        args.test_split_file or args.split_file, val_ann, "--test_split_file/--split_file"
    )
    if painted and not args.semantic_cache_dir:
        raise ValueError("--semantic_cache_dir is required for painted mode")
    config = build_pointpillars_config(
        kitti_root=args.kitti_root,
        train_ann_file=train_ann,
        val_ann_file=val_ann,
        semantic_cache_dir=args.semantic_cache_dir,
        work_dir=str(work_dir),
        epochs=args.epochs,
        lr=args.lr,
        painted=painted,
        base_config=args.base_config or args.point_cloud_cfg or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        submission_prefix=submission_prefix or None,
    )
    config.launcher = "none"
    return config


def _atomic_copy(source: Path, target: str) -> None:
    if not target:
        return
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".tmp")
    shutil.copyfile(str(source), str(temporary))
    os.replace(str(temporary), str(destination))


def _latest_epoch_checkpoint(work_dir: Path) -> Path:
    checkpoints = list(work_dir.glob("epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"no epoch checkpoint in {work_dir}")
    return max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))


def _register_modules() -> None:
    from mmdet3d.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    import model.pointpainting.mmdet3d_painted  # noqa: F401


def run_train(args) -> None:
    _register_modules()
    from mmengine.runner import Runner

    work_dir = Path(args.work_dir)
    if args.save_best_ckpt and (args.save_last_ckpt or args.save_ckpt):
        best_target = Path(args.save_best_ckpt).resolve()
        last_target = Path(args.save_last_ckpt or args.save_ckpt).resolve()
        if best_target == last_target:
            raise ValueError("best and last checkpoint targets must be different")
    config = _build_config(args, painted=True, work_dir=work_dir)
    if args.resume_ckpt and not args.reset_optimizer:
        config.resume = True
        config.load_from = args.resume_ckpt
    else:
        config.resume = False
        config.load_from = _require_file(
            args.resume_ckpt or args.fusion_ckpt or args.point_cloud_ckpt,
            "--resume_ckpt/--fusion_ckpt/--point_cloud_ckpt",
        )
    print(
        f"[PointPainting][train] painted=True epochs={args.epochs} lr={args.lr} "
        f"work_dir={work_dir} resume={config.resume}",
        flush=True,
    )
    runner = Runner.from_cfg(config)
    trainable = sum(parameter.numel() for parameter in runner.model.parameters() if parameter.requires_grad)
    print(f"[PointPainting][train] optimizer_param_count={trainable}", flush=True)
    runner.train()
    latest = _latest_epoch_checkpoint(work_dir)
    _atomic_copy(latest, args.save_last_ckpt or args.save_ckpt)
    if args.save_best_ckpt:
        best = runner.message_hub.get_info("best_ckpt")
        if not best or not Path(best).is_file():
            raise FileNotFoundError(f"best checkpoint was not produced: {best}")
        _atomic_copy(Path(best), args.save_best_ckpt)


def _run_test_once(args, painted: bool, checkpoint: str, output_dir: str, name: str):
    from mmengine.runner import Runner

    work_dir = Path(args.work_dir) / name
    if not output_dir:
        raise ValueError(f"output directory is required for {name}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _build_config(
        args,
        painted=painted,
        work_dir=work_dir,
        submission_prefix=str(output),
    )
    config.load_from = _require_file(checkpoint, f"{name} checkpoint")
    print(
        f"[PointPainting][test] name={name} painted={painted} checkpoint={checkpoint} "
        f"pred_dir={output}",
        flush=True,
    )
    return Runner.from_cfg(config).test()


def run_test(args) -> None:
    _register_modules()
    if args.save_raw_pred_dir:
        raw_metrics = _run_test_once(
            args,
            False,
            args.raw_ckpt or args.point_cloud_ckpt,
            args.save_raw_pred_dir,
            "raw_eval",
        )
        print(f"[PointPainting][test] raw_metrics={raw_metrics}", flush=True)
    if args.raw_only:
        if not args.save_raw_pred_dir:
            raise ValueError("--save_raw_pred_dir is required with --raw_only")
        return
    painted_metrics = _run_test_once(
        args,
        True,
        args.fusion_ckpt or args.resume_ckpt,
        args.pred_dir,
        "painted_eval",
    )
    print(f"[PointPainting][test] painted_metrics={painted_metrics}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
