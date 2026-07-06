# -*- coding: utf-8 -*-
"""
test_script.py (KITTI official AP version, aligned with revised CLOCS pipeline)

功能：
1) 用修改后的 CLOCSFramework 在 KITTI val 上跑预测
2) 写出 KITTI label_2 格式预测文件
3) 调用 mmdet3d 的 KITTI eval 输出：bbox / bev / 3d AP
4) 支持训练 FusionMLP（默认冻结 2D / 3D detector）

说明：
- 该版本与修改后的 build_clocs_system.py / clocs_framework.py / match_pair_module.py 对齐
- 默认按 KITTI Car 单类任务运行
- 当前仍是“post-NMS detector output + fusion rerank”版本
- 若要进一步逼近论文精度，后续还应改成 pre-NMS proposals
"""

from __future__ import annotations

import os
import argparse
import random
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

# 激活 fusionrepo 自定义算子注册
import utils  # noqa: F401

from model.clocs.initialization.build_clocs_system import (
    build_clocs_system,
    train_step,
    test_step,
)
from utils.pcdet_kitti_writer import write_kitti_pred_txt_pcdet

import common  # noqa: F401


def normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p


def list_ids(image_dir: str) -> List[str]:
    ids = []
    for fn in sorted(os.listdir(image_dir)):
        if fn.endswith(".png"):
            ids.append(os.path.splitext(fn)[0])
    return ids


def parse_int_list(s: str) -> Optional[Sequence[int]]:
    s = s.strip()
    if s == "":
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")


def cam_rect_to_lidar_points(
    pts_cam: np.ndarray,
    Tr_velo_to_cam: np.ndarray,
    R0_rect: np.ndarray
) -> np.ndarray:
    """
    camera rect -> lidar
    pts_cam: (N,3)
    """
    pts_cam = np.asarray(pts_cam, dtype=np.float32)
    N = pts_cam.shape[0]

    pts_h = np.concatenate([pts_cam, np.ones((N, 1), dtype=np.float32)], axis=1)

    R0_ext = np.eye(4, dtype=np.float32)
    R0_ext[:3, :3] = R0_rect

    V2C_ext = np.eye(4, dtype=np.float32)
    V2C_ext[:3, :4] = Tr_velo_to_cam

    pts_lidar = np.linalg.inv(V2C_ext) @ (np.linalg.inv(R0_ext) @ pts_h.T)
    return pts_lidar[:3, :].T


def read_kitti_calib(calib_path: str) -> Dict[str, np.ndarray]:
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            vals = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
            data[key] = vals

    P2 = data["P2"].reshape(3, 4)
    R0 = data["R0_rect"].reshape(3, 3)
    Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    return {"P2": P2, "R0_rect": R0, "Tr_velo_to_cam": Tr}


def read_kitti_label_as_lidar_boxes(
    label_path: str,
    calib: Dict[str, np.ndarray],
    only_car: bool = True
) -> np.ndarray:
    """
    读取 KITTI label_2，输出 LiDAR box:
    [x, y, z, w, l, h, yaw]
    与当前项目中的 3D box 约定保持一致
    """
    boxes = []

    if not os.path.exists(label_path):
        return np.zeros((0, 7), dtype=np.float32)

    Tr = calib["Tr_velo_to_cam"]
    R0 = calib["R0_rect"]

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            p = line.split()
            cls = p[0]
            if only_car and cls != "Car":
                continue

            h = float(p[8])
            w = float(p[9])
            l = float(p[10])

            x_cam = float(p[11])
            y_cam = float(p[12])
            z_cam = float(p[13])
            ry = float(p[14])

            # KITTI location 是 bottom center
            center_cam = np.array([[x_cam, y_cam - h / 2.0, z_cam]], dtype=np.float32)
            center_lidar = cam_rect_to_lidar_points(center_cam, Tr, R0)[0]

            # KITTI ry -> LiDAR yaw
            yaw_lidar = -ry - np.pi / 2.0

            boxes.append([
                center_lidar[0],
                center_lidar[1],
                center_lidar[2],
                w,
                l,
                h,
                yaw_lidar
            ])

    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    return np.array(boxes, dtype=np.float32)


def read_kitti_label_as_anno(label_path: str, only_car: bool = True) -> Dict[str, np.ndarray]:
    """
    读取 label_2，返回 mmdet3d kitti_eval 常用的 anno 字段
    坐标系：camera rect（KITTI 原生）
    """
    names = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dims = []      # h,w,l
    loc = []       # x,y,z (bottom center)
    ry = []

    if not os.path.exists(label_path):
        return dict(
            name=np.array([], dtype=object),
            truncated=np.array([], dtype=np.float32),
            occluded=np.array([], dtype=np.int64),
            alpha=np.array([], dtype=np.float32),
            bbox=np.zeros((0, 4), dtype=np.float32),
            dimensions=np.zeros((0, 3), dtype=np.float32),
            location=np.zeros((0, 3), dtype=np.float32),
            rotation_y=np.array([], dtype=np.float32),
        )

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            cls = p[0]
            if only_car and cls != "Car":
                continue

            names.append(cls)
            truncated.append(float(p[1]))
            occluded.append(int(p[2]))
            alpha.append(float(p[3]))
            bbox.append([float(p[4]), float(p[5]), float(p[6]), float(p[7])])

            h = float(p[8])
            w = float(p[9])
            l = float(p[10])
            dims.append([h, w, l])

            x = float(p[11])
            y = float(p[12])
            z = float(p[13])
            loc.append([x, y, z])

            ry.append(float(p[14]))

    return dict(
        name=np.array(names, dtype=object),
        truncated=np.array(truncated, dtype=np.float32),
        occluded=np.array(occluded, dtype=np.int64),
        alpha=np.array(alpha, dtype=np.float32),
        bbox=np.array(bbox, dtype=np.float32) if len(bbox) else np.zeros((0, 4), dtype=np.float32),
        dimensions=np.array(dims, dtype=np.float32) if len(dims) else np.zeros((0, 3), dtype=np.float32),
        location=np.array(loc, dtype=np.float32) if len(loc) else np.zeros((0, 3), dtype=np.float32),
        rotation_y=np.array(ry, dtype=np.float32),
    )


def run_kitti_eval(gt_annos, dt_annos, class_names=("Car",)):
    from mmdet3d.evaluation.functional.kitti_utils.eval import kitti_eval  # type: ignore

    result_str, result_dict = kitti_eval(
        gt_annos=gt_annos,
        dt_annos=dt_annos,
        current_classes=list(class_names),
        eval_types=["bbox", "bev", "3d"],
    )
    return {"report": result_str, "metrics": result_dict}


def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--kitti_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--mode", type=str, default="eval", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--split_file", type=str, default="", help="val split file")

    ap.add_argument(
        "--det2d_cfg",
        type=str,
        default="/root/miniconda3/envs/fusionrepo/lib/python3.8/site-packages/mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"
    )
    ap.add_argument(
        "--det3d_cfg",
        type=str,
        default="/root/miniconda3/envs/fusionrepo/lib/python3.8/site-packages/mmdet3d/.mim/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
    )
    ap.add_argument(
        "--det2d_ckpt",
        type=str,
        default="./checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth"
    )
    ap.add_argument(
        "--det3d_ckpt",
        type=str,
        default="./checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
    )

    # 关键：类别过滤，默认按 Car
    # COCO 中 car 常见为 2；KITTI 3D car 常见为 0
    ap.add_argument("--det2d_allowed_labels", type=str, default="2")
    ap.add_argument("--det3d_allowed_labels", type=str, default="0")

    ap.add_argument("--det2d_score_thr", type=float, default=0.05)
    ap.add_argument("--det2d_max_boxes", type=int, default=200)
    ap.add_argument("--det3d_score_thr", type=float, default=0.05)
    ap.add_argument("--det3d_max_boxes", type=int, default=300)

    ap.add_argument("--pair_iou_thr", type=float, default=0.1)
    ap.add_argument("--nms3d_thr", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--pred_dir", type=str, default="./pred_kitti")
    ap.add_argument("--pred_score_thr", type=float, default=0.0)
    ap.add_argument("--fusion_ckpt", type=str, default="./checkpoints/clocs_fusion_trained.pth")

    return ap


def load_split_ids(image_dir: str, all_ids: List[str], split_file: str, val_ratio: float, seed: int) -> List[str]:
    if split_file:
        split_path = normalize_path(split_file)
        with open(split_path, "r") as f:
            val_ids = [x.strip() for x in f.readlines() if x.strip()]
        val_ids = [i for i in val_ids if os.path.exists(os.path.join(image_dir, i + ".png"))]
        assert len(val_ids) > 0, f"split_file empty or not matched: {split_path}"
        return val_ids

    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    return sorted(ids[:n_val])


def main():
    ap = build_argparser()
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    kitti_root = normalize_path(args.kitti_root)
    train_root = os.path.join(kitti_root, "training")
    image_dir = os.path.join(train_root, "image_2")
    lidar_dir = os.path.join(train_root, "velodyne")
    calib_dir = os.path.join(train_root, "calib")
    label_dir = os.path.join(train_root, "label_2")

    assert os.path.isdir(image_dir), f"Not found: {image_dir}"
    assert os.path.isdir(lidar_dir), f"Not found: {lidar_dir}"
    assert os.path.isdir(calib_dir), f"Not found: {calib_dir}"
    assert os.path.isdir(label_dir), f"Not found: {label_dir}"

    ids = list_ids(image_dir)
    assert len(ids) > 0, "No png files found in image_2"

    val_ids = load_split_ids(image_dir, ids, args.split_file, args.val_ratio, args.seed)

    print(f"[KITTI] val={len(val_ids)} root={kitti_root}")
    print(f"[PRED] pred_dir={args.pred_dir}")

    det2d_allowed_labels = parse_int_list(args.det2d_allowed_labels)
    det3d_allowed_labels = parse_int_list(args.det3d_allowed_labels)

    sys = build_clocs_system(
        det2d_cfg=args.det2d_cfg,
        det2d_ckpt=normalize_path(args.det2d_ckpt),
        det3d_cfg=args.det3d_cfg,
        det3d_ckpt=normalize_path(args.det3d_ckpt),
        device=args.device,

        det2d_score_thr=args.det2d_score_thr,
        det2d_max_boxes=args.det2d_max_boxes,
        det3d_score_thr=args.det3d_score_thr,
        det3d_max_boxes=args.det3d_max_boxes,

        det2d_allowed_labels=det2d_allowed_labels,
        det3d_allowed_labels=det3d_allowed_labels,

        pair_iou_thr=args.pair_iou_thr,
        nms3d_thr=args.nms3d_thr,
        score_fuse_alpha=args.alpha,
        score_fuse_beta=args.beta,
        score_fuse_mode="weighted_sum",
        lr=args.lr,
        freeze_detectors=True,
    )

    fusion_ckpt = normalize_path(args.fusion_ckpt)
    if os.path.exists(fusion_ckpt):
        ckpt = torch.load(fusion_ckpt, map_location="cpu")
        sys.model.load_state_dict(ckpt, strict=False)
        print(f"[LOAD] fusion checkpoint: {fusion_ckpt}")
    else:
        print(f"[WARN] fusion checkpoint not found: {fusion_ckpt}")

    if args.mode == "train":
        val_set = set(val_ids)
        train_ids = [i for i in ids if i not in val_set]
        random.shuffle(train_ids)

        print(f"[TRAIN] num_train={len(train_ids)} epochs={args.epochs}")

        for ep in range(1, args.epochs + 1):
            losses = []

            for idx, sid in enumerate(train_ids, start=1):
                img_path = os.path.join(image_dir, sid + ".png")
                lidar_path = os.path.join(lidar_dir, sid + ".bin")
                calib_path = os.path.join(calib_dir, sid + ".txt")
                label_path = os.path.join(label_dir, sid + ".txt")

                calib = read_kitti_calib(calib_path)
                gt_boxes_lidar = read_kitti_label_as_lidar_boxes(label_path, calib, only_car=True)

                if gt_boxes_lidar.shape[0] == 0:
                    continue

                # 单类 Car 训练，GT label 全置 0
                gt_labels_3d = np.zeros((gt_boxes_lidar.shape[0],), dtype=np.int64)

                batch = {
                    "image": img_path,
                    "images": img_path,
                    "point": lidar_path,
                    "points": lidar_path,
                    "image_path": img_path,
                    "lidar_path": lidar_path,
                    "calib": calib,
                    "gt_boxes_3d": torch.from_numpy(gt_boxes_lidar).float(),
                    "gt_labels_3d": torch.from_numpy(gt_labels_3d).long(),
                }

                out = train_step(sys, batch)
                losses.append(float(out["loss_fusion"].item()))

                if idx % 50 == 0:
                    mean_loss = sum(losses) / max(len(losses), 1)
                    print(f"[TRAIN] epoch={ep} iter={idx}/{len(train_ids)} loss_mean={mean_loss:.6f}")

            mean_loss = sum(losses) / max(len(losses), 1)
            print(f"[TRAIN] epoch={ep} loss_mean={mean_loss:.6f}")

        os.makedirs("./checkpoints", exist_ok=True)
        save_path = "./checkpoints/clocs_fusion_trained.pth"
        torch.save(sys.model.state_dict(), save_path)
        print(f"[SAVE] {save_path}")
        return

    # ---------------- EVAL ----------------
    gt_annos: List[Dict[str, Any]] = []
    dt_annos: List[Dict[str, Any]] = []

    os.makedirs(args.pred_dir, exist_ok=True)

    for idx, sid in enumerate(val_ids, start=1):
        img_path = os.path.join(image_dir, sid + ".png")
        lidar_path = os.path.join(lidar_dir, sid + ".bin")
        calib_path = os.path.join(calib_dir, sid + ".txt")
        label_path = os.path.join(label_dir, sid + ".txt")

        calib = read_kitti_calib(calib_path)

        im = Image.open(img_path)
        W, H = im.size
        image_shape = (H, W)

        pred = test_step(sys, {
            "image": img_path,
            "images": img_path,
            "point": lidar_path,
            "points": lidar_path,
            "calib": calib
        })

        boxes_lidar = pred["boxes_3d"].detach().cpu()
        scores = pred["scores_3d"].detach().cpu()

        if idx <= 5:
            pair_dict = pred.get("pair_dict", {})
            num_pairs = 0
            if isinstance(pair_dict, dict) and "pair_indices" in pair_dict:
                num_pairs = int(pair_dict["pair_indices"].shape[0])
            print(
                f"[DEBUG][{sid}] "
                f"num_pred={boxes_lidar.shape[0]} "
                f"num_pairs={num_pairs}"
            )
        calib_path = f"/root/kitti/training/calib/{sid}.txt"
        write_kitti_pred_txt_pcdet(
            pred_dir=args.pred_dir,
            frame_id=sid,
            boxes_lidar=boxes_lidar,
            scores=scores,
            calib_path=calib_path,
            image_shape=image_shape,
            cls_name="Car",
            score_thr=args.pred_score_thr
        )

        gt = read_kitti_label_as_anno(label_path, only_car=True)
        gt_annos.append(gt)

        dt_path = os.path.join(args.pred_dir, "data", sid + ".txt")
        dt = read_kitti_label_as_anno(dt_path, only_car=True)

        scores_list = []
        names_list = []
        if os.path.exists(dt_path):
            with open(dt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = line.split()
                    names_list.append(p[0])
                    scores_list.append(float(p[-1]))

        dt["name"] = np.array(names_list, dtype=object)
        dt["score"] = np.array(scores_list, dtype=np.float32)
        dt_annos.append(dt)

        if idx % 50 == 0:
            print(f"[EVAL] processed {idx}/{len(val_ids)}")

    print("[EVAL] running KITTI official eval (Car, IoU=0.7)...")
    res = run_kitti_eval(gt_annos, dt_annos, class_names=("Car",))

    if "report" in res:
        print(res["report"])
    else:
        print(res)

    print(f"[DONE] pred files saved to: {args.pred_dir}")


if __name__ == "__main__":
    main()