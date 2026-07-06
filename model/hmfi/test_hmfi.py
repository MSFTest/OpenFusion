# -*- coding: utf-8 -*-
"""
test_hmfi.py (HMFI KITTI train/test unified version)

功能：
1) mode=test:
   - 在 KITTI val 上调用 HMFI 做预测
   - 写出 KITTI label_2 格式预测文件
   - 调用 mmdet3d 的 KITTI eval 输出 bbox / bev / 3d AP

2) mode=train:
   - 使用 build_hmfi_system + train_step 做真实训练
   - 读取 KITTI train split
   - 构造 gt_boxes / gt_labels
   - 每轮保存 checkpoint
"""

from __future__ import annotations

import os
import argparse
import random
from typing import Dict, Any, List, Tuple

# 激活 fusionrepo 自定义算子注册
import utils  # noqa: F401
import common  # noqa: F401
import fusion_unit  # noqa: F401

import numpy as np
import torch
from PIL import Image
import math
import cv2

from model.hmfi.initialization.build_hmfi_system import (
    build_hmfi_system,
    train_step,
    test_step,
)


# ============================================================
# 1) 基础工具
# ============================================================
def normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p


def pad_to_divisor(img: torch.Tensor, size_divisor: int = 32) -> torch.Tensor:
    """
    img: (C,H,W)
    return: (C,H_pad,W_pad)
    """
    c, h, w = img.shape
    pad_h = int(math.ceil(h / size_divisor) * size_divisor - h)
    pad_w = int(math.ceil(w / size_divisor) * size_divisor - w)

    if pad_h == 0 and pad_w == 0:
        return img

    out = torch.zeros((c, h + pad_h, w + pad_w), dtype=img.dtype)
    out[:, :h, :w] = img
    return out


def load_image_for_mmdet_caffe(img_path: str, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    按 faster-rcnn_r50-caffe_fpn_ms-1x_coco 的常见预处理方式做：
    - BGR
    - float32
    - mean=[103.53,116.28,123.675]
    - std=[1,1,1]
    - pad 到 32 的倍数

    返回：
        image_tensor: (1,3,H_pad,W_pad)
        meta: 原始/填充尺寸信息
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    img = img.astype(np.float32)

    ori_h, ori_w = img.shape[:2]

    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    img = (img - mean) / std

    # HWC -> CHW
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W)

    img = pad_to_divisor(img, size_divisor=32)
    pad_h, pad_w = img.shape[1], img.shape[2]

    img = img.unsqueeze(0).to(device)  # (1,3,H,W)

    meta = {
        "ori_shape": (ori_h, ori_w),
        "img_shape": (ori_h, ori_w),
        "pad_shape": (pad_h, pad_w),
        "scale_factor": 1.0,
        "flip": False,
    }
    return img, meta


def load_points_tensor(lidar_path: str, device: str) -> torch.Tensor:
    """
    KITTI velodyne bin -> (N,4) float tensor
    """
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    pts = torch.from_numpy(pts).float().to(device)
    return pts


def make_mmdet3d_train_sample(gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
    from mmengine.structures import InstanceData
    from mmdet3d.structures import Box3DMode, Det3DDataSample, LiDARInstance3DBoxes

    ds = Det3DDataSample()
    gt_instances_3d = InstanceData()
    gt_instances_3d.bboxes_3d = LiDARInstance3DBoxes(gt_boxes, box_dim=7)
    gt_instances_3d.labels_3d = gt_labels.long()
    ds.gt_instances_3d = gt_instances_3d
    ds.set_metainfo({
        "box_type_3d": LiDARInstance3DBoxes,
        "box_mode_3d": Box3DMode.LIDAR,
    })
    return ds


def apply_safe_train_augmentation(
    points: torch.Tensor,
    gt_boxes: torch.Tensor,
    enable_point_shuffle: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal-safe augmentation.

    We only shuffle point order here. Geometric LiDAR augmentations such as
    rotation/scale/flip must update image projection geometry as well; applying
    them in this hand-built batch path would break camera-LiDAR alignment.
    """
    if enable_point_shuffle and points.numel() > 0:
        perm = torch.randperm(points.shape[0], device=points.device)
        points = points[perm]
    return points, gt_boxes


def list_ids(image_dir: str) -> List[str]:
    ids = []
    for fn in sorted(os.listdir(image_dir)):
        if fn.endswith(".png"):
            ids.append(os.path.splitext(fn)[0])
    return ids


def read_split_file(split_file: str) -> List[str]:
    with open(split_file, "r") as f:
        ids = [x.strip() for x in f.readlines() if x.strip()]
    return ids


def parse_float_list(value: str, expected_len: int, name: str) -> List[float]:
    vals = [float(x) for x in value.split(",") if x.strip()]
    if len(vals) != expected_len:
        raise ValueError(f"{name} expects {expected_len} comma-separated values, got {len(vals)}: {value}")
    return vals


# ============================================================
# 2) KITTI 标定 / 标签读取
# ============================================================
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

    return {
        "P2": P2,
        "R0_rect": R0,
        "Tr_velo_to_cam": Tr
    }


def read_kitti_label_as_anno(label_path: str, only_car: bool = True) -> Dict[str, np.ndarray]:
    """
    读取 label_2，返回 mmdet3d kitti_eval 常用 anno 字段
    坐标系：camera rect（KITTI 原生）
    """
    names = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dims = []      # (N,3) h,w,l
    loc = []       # (N,3) x,y,z (bottom center in camera rect)
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


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _limit_period_np(x: np.ndarray, period: float = 2 * np.pi, offset: float = 0.5) -> np.ndarray:
    return x - np.floor(x / period + offset) * period


def write_kitti_pred_txt_hmfi(
    pred_dir: str,
    frame_id: str,
    boxes_lidar,
    scores,
    calib_path: str,
    image_shape,
    cls_name: str = "Car",
    score_thr: float = 0.0
):
    """
    HMFI 专用 KITTI 写出。

    HMFI/Part-A2 输出为 mmdet3d 标准 LiDAR 中心框:
        [x, y, z, dx, dy, dz, yaw]
    OpenPCDet 的 boxes3d_lidar_to_kitti_camera 会负责转成 KITTI
    camera bottom-center 格式，因此这里不能再手动 z += dz / 2。
    """
    from pcdet.utils.calibration_kitti import Calibration
    from pcdet.utils.box_utils import (
        boxes3d_lidar_to_kitti_camera,
        boxes3d_kitti_camera_to_imageboxes,
    )

    os.makedirs(pred_dir, exist_ok=True)
    data_dir = os.path.join(pred_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{frame_id}.txt")

    boxes_lidar = _to_numpy(boxes_lidar).astype(np.float32)
    scores = _to_numpy(scores).astype(np.float32).reshape(-1)

    if boxes_lidar.shape[0] > 0:
        keep = scores >= float(score_thr)
        boxes_lidar = boxes_lidar[keep]
        scores = scores[keep]

    if boxes_lidar.shape[0] == 0:
        open(out_path, "w").close()
        return out_path

    boxes_pcdet = boxes_lidar.copy()
    calib_obj = Calibration(calib_path)

    boxes_camera = boxes3d_lidar_to_kitti_camera(boxes_pcdet, calib_obj)
    boxes_img = boxes3d_kitti_camera_to_imageboxes(
        boxes_camera,
        calib_obj,
        image_shape=image_shape,
    )

    alpha = boxes_camera[:, 6] - np.arctan2(boxes_camera[:, 0], boxes_camera[:, 2])
    alpha = _limit_period_np(alpha)
    ry = _limit_period_np(boxes_camera[:, 6])

    H, W = int(image_shape[0]), int(image_shape[1])
    lines = []
    for i in range(boxes_camera.shape[0]):
        x1, y1, x2, y2 = boxes_img[i]
        x1 = np.clip(x1, 0, W - 1)
        x2 = np.clip(x2, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        y2 = np.clip(y2, 0, H - 1)

        x, y, z = boxes_camera[i, 0:3]
        l, h, w = boxes_camera[i, 3:6]
        score = scores[i]

        if z <= 0:
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        lines.append(
            f"{cls_name} 0 0 {alpha[i]:.6f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.6f} {w:.6f} {l:.6f} "
            f"{x:.6f} {y:.6f} {z:.6f} "
            f"{ry[i]:.6f} {score:.6f}"
        )

    with open(out_path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")

    return out_path


def make_transform_mats(calib: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
        lidar_to_rect: (4,4)
        rect_to_lidar: (4,4)
    """
    R0 = calib["R0_rect"]
    Tr = calib["Tr_velo_to_cam"]

    R0_4 = np.eye(4, dtype=np.float32)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4, dtype=np.float32)
    Tr_4[:3, :4] = Tr

    lidar_to_rect = R0_4 @ Tr_4
    rect_to_lidar = np.linalg.inv(lidar_to_rect)
    return lidar_to_rect, rect_to_lidar


def rect_points_to_lidar(points_rect: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    """
    points_rect: (N,3)
    return: (N,3)
    """
    _, rect_to_lidar = make_transform_mats(calib)
    pts_h = np.concatenate(
        [points_rect.astype(np.float32), np.ones((points_rect.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    pts_lidar = (rect_to_lidar @ pts_h.T).T[:, :3]
    return pts_lidar


def read_kitti_gt_boxes_lidar(
    label_path: str,
    calib: Dict[str, np.ndarray],
    only_car: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 KITTI label_2 读取 GT，并转换到 lidar 坐标系。
    输出 boxes_lidar: (N,7) [x,y,z,dx,dy,dz,yaw]
    这里采用常见 KITTI->LiDAR 转换约定：
    - camera label 中 location 为底面中心
    - 转到 lidar 后 z 调整为盒中心
    - yaw_lidar = -(pi/2 + ry_camera)
    """
    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

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
            x = float(p[11])
            y = float(p[12])
            z = float(p[13])
            ry = float(p[14])

            loc_rect = np.array([[x, y, z]], dtype=np.float32)
            loc_lidar = rect_points_to_lidar(loc_rect, calib)[0]

            # KITTI camera label 常用为底面中心；转 lidar 后改成几何中心
            loc_lidar[2] += h / 2.0

            yaw_lidar = -(np.pi / 2.0 + ry)

            # [x,y,z,dx,dy,dz,yaw] 其中 dx=l, dy=w, dz=h
            boxes.append([
                float(loc_lidar[0]),
                float(loc_lidar[1]),
                float(loc_lidar[2]),
                float(l),
                float(w),
                float(h),
                float(yaw_lidar),
            ])
            labels.append(1)  # Car -> 1

    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


# ============================================================
# 3) eval 导入
# ============================================================
def run_kitti_eval(gt_annos, dt_annos, class_names=("Car",)):
    from mmdet3d.evaluation.functional.kitti_utils.eval import kitti_eval  # type: ignore

    current_classes = list(class_names)
    result_str, result_dict = kitti_eval(
        gt_annos=gt_annos,
        dt_annos=dt_annos,
        current_classes=current_classes,
        eval_types=["bbox", "bev", "3d"],
    )
    return {
        "report": result_str,
        "metrics": result_dict
    }


# ============================================================
# 4) 构造 batch
# ============================================================
def build_eval_batch(
    img_path: str,
    lidar_path: str,
    calib: Dict[str, np.ndarray],
    device: str
) -> Dict[str, Any]:
    """
    构造 HMFI 单样本测试 batch
    """
    image_tensor, image_meta = load_image_for_mmdet_caffe(img_path, device)
    points_tensor = load_points_tensor(lidar_path, device)

    proj_matrix = torch.from_numpy(calib["P2"]).float().to(device)

    # 构建 lidar_to_cam 变换矩阵
    R0 = calib["R0_rect"]
    Tr = calib["Tr_velo_to_cam"]
    R0_4 = np.eye(4, dtype=np.float32)
    R0_4[:3, :3] = R0
    Tr_4 = np.eye(4, dtype=np.float32)
    Tr_4[:3, :4] = Tr
    lidar_to_cam = R0_4 @ Tr_4  # LiDAR to rect camera

    lidar_to_cam_tensor = torch.from_numpy(lidar_to_cam).float().to(device)

    batch = {
        "images": image_tensor,
        "points": [points_tensor],
        "image_path": img_path,
        "lidar_path": lidar_path,
        "calib": calib,
        "proj_matrix": proj_matrix,
        "lidar_to_cam": lidar_to_cam_tensor,
        "cam_intrinsic": proj_matrix[:3, :3],
        "img_metas": [
            {
                "img_path": img_path,
                "lidar_path": lidar_path,
                "ori_shape": image_meta["ori_shape"],
                "img_shape": image_meta["img_shape"],
                "pad_shape": image_meta["pad_shape"],
                "scale_factor": image_meta["scale_factor"],
                "flip": image_meta["flip"],
            }
        ],
    }
    return batch


def build_train_batch(
    img_path: str,
    lidar_path: str,
    calib_path: str,
    label_path: str,
    device: str,
    enable_point_shuffle: bool = True
) -> Dict[str, Any]:
    """
    构造 HMFI 单样本训练 batch
    """
    calib = read_kitti_calib(calib_path)
    image_tensor, image_meta = load_image_for_mmdet_caffe(img_path, device)
    points_tensor = load_points_tensor(lidar_path, device)

    gt_boxes_np, gt_labels_np = read_kitti_gt_boxes_lidar(
        label_path=label_path,
        calib=calib,
        only_car=True
    )

    gt_boxes = torch.from_numpy(gt_boxes_np).float().to(device)
    gt_labels = torch.from_numpy(gt_labels_np).long().to(device)
    points_tensor, gt_boxes = apply_safe_train_augmentation(
        points_tensor,
        gt_boxes,
        enable_point_shuffle=enable_point_shuffle
    )

    proj_matrix = torch.from_numpy(calib["P2"]).float().to(device)

    # 构建 lidar_to_cam 变换矩阵
    R0 = calib["R0_rect"]
    Tr = calib["Tr_velo_to_cam"]
    R0_4 = np.eye(4, dtype=np.float32)
    R0_4[:3, :3] = R0
    Tr_4 = np.eye(4, dtype=np.float32)
    Tr_4[:3, :4] = Tr
    lidar_to_cam = R0_4 @ Tr_4  # LiDAR to rect camera

    lidar_to_cam_tensor = torch.from_numpy(lidar_to_cam).float().to(device)

    batch = {
        "images": image_tensor,
        "points": [points_tensor],
        "image_path": img_path,
        "lidar_path": lidar_path,
        "calib": calib,
        "proj_matrix": proj_matrix,
        "lidar_to_cam": lidar_to_cam_tensor,
        "cam_intrinsic": proj_matrix[:3, :3],
        "gt_boxes": [gt_boxes],
        "gt_labels": [gt_labels],
        "inputs": {"points": [points_tensor]},
        "data_samples": [make_mmdet3d_train_sample(gt_boxes, gt_labels)],
        "img_metas": [
            {
                "img_path": img_path,
                "lidar_path": lidar_path,
                "ori_shape": image_meta["ori_shape"],
                "img_shape": image_meta["img_shape"],
                "pad_shape": image_meta["pad_shape"],
                "scale_factor": image_meta["scale_factor"],
                "flip": image_meta["flip"],
            }
        ],
    }
    return batch


# ============================================================
# 5) ckpt 工具
# ============================================================
def load_fusion_checkpoint(
    sys,
    ckpt_path: str,
    for_test: bool = False,
    resume_optimizer: bool = False
):
    ckpt_path = normalize_path(ckpt_path)
    if not ckpt_path:
        if for_test:
            print("[WARN] fusion checkpoint path empty.")
        return

    if not os.path.exists(ckpt_path):
        if for_test:
            print(f"[WARN] fusion checkpoint not found: {ckpt_path}")
            print("[WARN] 将直接评测当前初始化权重，结果通常不会正常。")
        else:
            print(f"[WARN] resume checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = sys.model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] checkpoint: {ckpt_path}")
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    if resume_optimizer and isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            sys.optimizer.load_state_dict(ckpt["optimizer"])
            print("[LOAD] optimizer state restored.")
        except Exception as e:
            print(f"[WARN] optimizer restore failed: {e}")
    elif (not for_test) and isinstance(ckpt, dict) and "optimizer" in ckpt:
        print("[LOAD] optimizer state not restored.")

    meta = {}
    if isinstance(ckpt, dict):
        for key in ("epoch", "best_loss", "best_epoch"):
            if key in ckpt:
                meta[key] = ckpt[key]
    return meta


def save_train_checkpoint(
    sys,
    ckpt_path: str,
    epoch: int,
    best_loss: float = None,
    best_epoch: int = None
):
    ckpt_path = normalize_path(ckpt_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "state_dict": sys.model.state_dict(),
        "optimizer": sys.optimizer.state_dict(),
    }
    if best_loss is not None:
        ckpt["best_loss"] = float(best_loss)
    if best_epoch is not None:
        ckpt["best_epoch"] = int(best_epoch)
    torch.save(ckpt, ckpt_path)
    print(f"[SAVE] checkpoint saved to: {ckpt_path}")


def auto_select_resume_checkpoint(save_ckpt: str) -> str:
    """
    Pick a checkpoint for continuing training when --resume_ckpt is omitted.

    Prefer the last checkpoint because it contains the latest optimizer state.
    The *_best checkpoint is for evaluation/model selection, not normal resume.
    """
    save_ckpt = normalize_path(save_ckpt)
    best_ckpt = save_ckpt.replace(".pth", "_best.pth")

    if os.path.exists(save_ckpt):
        return save_ckpt
    if os.path.exists(best_ckpt):
        return best_ckpt
    return ""


# ============================================================
# 6) train / test 主流程
# ============================================================
def run_test(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids):
    # -------------------------
    # val split
    # -------------------------
    if args.split_file:
        split_path = normalize_path(args.split_file)
        val_ids = read_split_file(split_path)
        val_ids = [i for i in val_ids if os.path.exists(os.path.join(image_dir, i + ".png"))]
        assert len(val_ids) > 0, f"split_file empty or not matched: {split_path}"
    else:
        ids_copy = list(ids)
        random.shuffle(ids_copy)
        n_val = int(len(ids_copy) * args.val_ratio)
        val_ids = sorted(ids_copy[:n_val])

    print(f"[KITTI] val={len(val_ids)}  root={kitti_root}")
    print(f"[PRED] pred_dir={args.pred_dir}")

    # -------------------------
    # 加载训练好的 HMFI 权重
    # -------------------------
    load_fusion_checkpoint(sys, args.fusion_ckpt, for_test=True)

    # -------------------------
    # EVAL: write preds + official eval
    # -------------------------
    gt_annos: List[Dict[str, Any]] = []
    dt_annos: List[Dict[str, Any]] = []

    os.makedirs(args.pred_dir, exist_ok=True)

    for idx, sid in enumerate(val_ids, start=1):
        img_path = os.path.join(image_dir, sid + ".png")
        lidar_path = os.path.join(lidar_dir, sid + ".bin")
        calib_path = os.path.join(calib_dir, sid + ".txt")
        label_path = os.path.join(label_dir, sid + ".txt")

        calib = read_kitti_calib(calib_path)

        # image shape
        im = Image.open(img_path)
        W, H = im.size
        image_shape = (H, W)

        # 1) 跑 HMFI
        batch = build_eval_batch(
            img_path=img_path,
            lidar_path=lidar_path,
            calib=calib,
            device=args.device
        )
        out = test_step(sys, batch)

        final_results = out["final_results"]
        if len(final_results) == 0:
            boxes_lidar = torch.zeros((0, 7), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
        else:
            pred0 = final_results[0]
            boxes_lidar = pred0["boxes_3d"].detach().cpu()
            scores = pred0["scores_3d"].detach().cpu()

        # 2) 写 KITTI txt
        write_kitti_pred_txt_hmfi(
            pred_dir=args.pred_dir,
            frame_id=sid,
            boxes_lidar=boxes_lidar,
            scores=scores,
            calib_path=calib_path,
            image_shape=image_shape,
            cls_name="Car",
            score_thr=args.pred_score_thr
        )

        # 3) 准备 eval 所需 annos
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

        if idx % 50 == 0 or idx == len(val_ids):
            print(f"[EVAL] processed {idx}/{len(val_ids)}")

    print("[EVAL] running KITTI official eval (Car, IoU=0.7, 40 recall positions)...")
    try:
        res = run_kitti_eval(gt_annos, dt_annos, class_names=("Car",))
        print("\n================ KITTI EVAL REPORT ================\n")
        print(res["report"])
        print("===================================================\n")
    except IndexError as e:
        print(f"[WARN] mmdet3d KITTI eval skipped because empty predictions triggered: {e}")
        print("[WARN] prediction txt files were still written; use KITTI native evaluation on pred_dir/data.")
    print(f"[DONE] prediction txt saved to: {os.path.join(args.pred_dir, 'data')}")


def run_train(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids):
    # -------------------------
    # train split
    # -------------------------
    if args.train_split_file:
        split_path = normalize_path(args.train_split_file)
    else:
        split_path = os.path.join(kitti_root, "ImageSets", "train.txt")

    train_ids = read_split_file(split_path)
    train_ids = [i for i in train_ids if os.path.exists(os.path.join(image_dir, i + ".png"))]
    assert len(train_ids) > 0, f"train split empty or not matched: {split_path}"

    print(f"[KITTI] train={len(train_ids)}  root={kitti_root}")
    print(f"[TRAIN] save_ckpt={args.save_ckpt}")

    best_loss = 1e18
    best_epoch = 0

    resume_ckpt = normalize_path(args.resume_ckpt) if args.resume_ckpt else auto_select_resume_checkpoint(args.save_ckpt)
    if resume_ckpt:
        print(f"[AUTO RESUME] loading checkpoint: {resume_ckpt}")
        meta = load_fusion_checkpoint(
            sys,
            resume_ckpt,
            for_test=False,
            resume_optimizer=args.resume_optimizer
        ) or {}
        if args.resume_best_loss and "best_loss" in meta:
            best_loss = float(meta["best_loss"])
            best_epoch = int(meta.get("best_epoch", meta.get("epoch", 0)))
            print(f"[AUTO RESUME] restored best_loss={best_loss:.6f} best_epoch={best_epoch}")
        else:
            print("[AUTO RESUME] best-loss tracking starts fresh for this run.")

    steps_per_epoch = int(math.ceil(len(train_ids) / max(args.batch_size, 1)))
    if args.use_onecycle:
        sys.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            sys.optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor
        )

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_ids)

        loss_sum = 0.0
        loss_rpn_sum = 0.0
        loss_feat_sum = 0.0
        loss_conf_sum = 0.0
        loss_reg_sum = 0.0
        num_pos_sum = 0.0
        num_neg_sum = 0.0
        step_count = 0

        print(f"\n================ Epoch {epoch}/{args.epochs} ================")

        sys.optimizer.zero_grad(set_to_none=True)

        for idx, sid in enumerate(train_ids, start=1):
            img_path = os.path.join(image_dir, sid + ".png")
            lidar_path = os.path.join(lidar_dir, sid + ".bin")
            calib_path = os.path.join(calib_dir, sid + ".txt")
            label_path = os.path.join(label_dir, sid + ".txt")

            batch = build_train_batch(
                img_path=img_path,
                lidar_path=lidar_path,
                calib_path=calib_path,
                label_path=label_path,
                device=args.device,
                enable_point_shuffle=args.enable_point_shuffle
            )

            # Forward
            sys.model.train()
            pred_out = sys.model.train_forward(batch)
            if "pc_out" in pred_out and "batch_inputs_dict" in pred_out["pc_out"]:
                batch["inputs"] = pred_out["pc_out"]["batch_inputs_dict"]

            from model.hmfi.initialization.build_hmfi_system import _compute_base_rpn_loss
            base_rpn_loss = _compute_base_rpn_loss(sys, batch)

            loss_dict = sys.criterion(
                pred_out=pred_out,
                batch=batch,
                base_rpn_loss=base_rpn_loss
            )

            loss = loss_dict["loss"]
            (loss / max(args.batch_size, 1)).backward()

            # 每 batch_size 个样本更新一次
            if idx % args.batch_size == 0:
                # 梯度裁剪
                if sys.grad_clip_norm is not None and sys.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(sys.model.parameters(), max_norm=sys.grad_clip_norm)

                sys.optimizer.step()
                if sys.scheduler is not None:
                    sys.scheduler.step()
                sys.optimizer.zero_grad(set_to_none=True)

                step_count += 1
                loss_sum += loss_dict["loss"].item()
                loss_rpn_sum += loss_dict["loss_rpn"].item()
                loss_feat_sum += loss_dict["loss_feat"].item()
                loss_conf_sum += loss_dict["loss_conf"].item()
                loss_reg_sum += loss_dict["loss_reg"].item()
                num_pos_sum += loss_dict["num_pos"].item()
                num_neg_sum += loss_dict["num_neg"].item()

                if step_count % args.log_interval == 0 or idx == len(train_ids):
                    print(
                        f"[TRAIN] epoch={epoch} step={idx}/{len(train_ids)} "
                        f"loss={loss_dict['loss'].item():.6f} "
                        f"loss_rpn={loss_dict['loss_rpn'].item():.6f} "
                        f"loss_conf={loss_dict['loss_conf'].item():.6f} "
                        f"loss_reg={loss_dict['loss_reg'].item():.6f} "
                        f"loss_vfim={loss_dict.get('loss_vfim', loss_dict['loss_feat']).item():.6f} "
                        f"num_rois={loss_dict['num_rois'].item():.0f} "
                        f"num_valid={loss_dict['num_valid'].item():.0f} "
                        f"num_pos={loss_dict['num_pos'].item():.0f} "
                        f"num_neg={loss_dict['num_neg'].item():.0f}"
                    )

        # 处理剩余未更新的梯度
        if len(train_ids) % args.batch_size != 0:
            if sys.grad_clip_norm is not None and sys.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(sys.model.parameters(), max_norm=sys.grad_clip_norm)
            sys.optimizer.step()
            if sys.scheduler is not None:
                sys.scheduler.step()
            sys.optimizer.zero_grad(set_to_none=True)
            step_count += 1

        mean_loss = loss_sum / max(step_count, 1)
        mean_rpn = loss_rpn_sum / max(step_count, 1)
        mean_feat = loss_feat_sum / max(step_count, 1)
        mean_conf = loss_conf_sum / max(step_count, 1)
        mean_reg = loss_reg_sum / max(step_count, 1)
        mean_pos = num_pos_sum / max(step_count, 1)
        mean_neg = num_neg_sum / max(step_count, 1)

        print(
            f"[EPOCH END] epoch={epoch} "
            f"mean_loss={mean_loss:.6f} "
            f"mean_loss_rpn={mean_rpn:.6f} "
            f"mean_loss_feat={mean_feat:.6f} "
            f"mean_loss_conf={mean_conf:.6f} "
            f"mean_loss_reg={mean_reg:.6f} "
            f"mean_num_pos={mean_pos:.2f} "
            f"mean_num_neg={mean_neg:.2f}"
        )

        # 保存 last
        save_train_checkpoint(
            sys,
            args.save_ckpt,
            epoch,
            best_loss=best_loss,
            best_epoch=best_epoch
        )

        # 保存 best
        best_ckpt = args.save_ckpt.replace(".pth", "_best.pth")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            save_train_checkpoint(
                sys,
                best_ckpt,
                epoch,
                best_loss=best_loss,
                best_epoch=best_epoch
            )
            save_train_checkpoint(
                sys,
                args.save_ckpt,
                epoch,
                best_loss=best_loss,
                best_epoch=best_epoch
            )


# ============================================================
# 7) 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # -------------------------
    # 模式
    # -------------------------
    ap.add_argument("--mode", type=str, default="test", choices=["train", "test"])

    # -------------------------
    # 基本参数
    # -------------------------
    ap.add_argument("--kitti_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--split_file", type=str, default="", help="test 模式下的 val split file")
    ap.add_argument("--train_split_file", type=str, default="", help="train 模式下的 train split file")
    ap.add_argument("--pred_dir", type=str, default="./pred_kitti_hmfi")
    ap.add_argument("--pred_score_thr", type=float, default=0.0)

    # -------------------------
    # 训练参数
    # -------------------------
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_ckpt", type=str, default="./checkpoints/hmfi_fusion_trained.pth")
    ap.add_argument("--resume_ckpt", type=str, default="")
    ap.add_argument("--resume_optimizer", action="store_true", default=False)
    ap.add_argument("--resume_best_loss", action="store_true", default=False)

    # -------------------------
    # 2D / 3D 后端配置
    # -------------------------
    ap.add_argument(
        "--image_cfg",
        type=str,
        default="/root/miniconda3/envs/fusionrepo/lib/python3.8/site-packages/mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"
    )
    ap.add_argument(
        "--point_cloud_cfg",
        type=str,
        default="/root/miniconda3/envs/fusionrepo/lib/python3.8/site-packages/mmdet3d/.mim/configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car.py"
    )
    ap.add_argument("--image_ckpt", type=str, default="./checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth")
    ap.add_argument("--point_cloud_ckpt", type=str, default="./checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth")
    ap.add_argument("--fusion_ckpt", type=str, default="./checkpoints/hmfi_fusion_trained.pth")

    # -------------------------
    # build_hmfi_system 参数
    # -------------------------
    ap.add_argument("--image_feat_index", type=int, default=0)
    ap.add_argument("--point_cloud_feat_index", type=int, default=-1)

    ap.add_argument("--lidar_channels", type=int, default=4)
    ap.add_argument("--image_channels", type=int, default=256)
    ap.add_argument("--fusion_out_channels", type=int, default=64)
    ap.add_argument("--d_k", type=int, default=64)
    ap.add_argument("--d_v", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--depth_mode", type=int, default=1)
    ap.add_argument("--downsample_factor", type=int, default=4)
    ap.add_argument("--depth_range", type=str, default="2.0,70.0")
    ap.add_argument("--num_depth_bins", type=int, default=80)
    ap.add_argument("--voxel_size", type=str, default="0.05,0.05,0.1")
    ap.add_argument("--point_cloud_range", type=str, default="0.0,-40.0,-3.0,70.4,40.0,1.0")
    ap.add_argument("--use_vfim", action="store_true", default=False)
    ap.add_argument("--disable_vfim", action="store_false", dest="use_vfim")
    ap.add_argument("--vfim_loss_weight", type=float, default=0.02)
    ap.add_argument("--enable_point_shuffle", action="store_true", default=True)
    ap.add_argument("--disable_point_shuffle", action="store_false", dest="enable_point_shuffle")
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--pos_iou_thr", type=float, default=0.45)
    ap.add_argument("--neg_iou_thr", type=float, default=0.20)
    ap.add_argument("--code_weights", type=str, default="1,1,2,1,1,2,2")
    ap.add_argument("--max_pos_weight", type=float, default=10.0)

    ap.add_argument("--score_thr", type=float, default=0.05)
    ap.add_argument("--nms_thr", type=float, default=0.1)
    ap.add_argument("--pre_nms_topk", type=int, default=4096)
    ap.add_argument("--post_nms_topk", type=int, default=500)
    ap.add_argument("--nms_sweep", action="store_true", default=False)
    ap.add_argument("--score_thr_sweep", type=str, default="0.01,0.05,0.1")
    ap.add_argument("--nms_thr_sweep", type=str, default="0.01,0.05,0.1")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip_norm", type=float, default=10.0)
    ap.add_argument("--debug_base_loss", action="store_true", default=False)
    ap.add_argument("--use_onecycle", action="store_true", default=False)
    ap.add_argument("--disable_onecycle", action="store_false", dest="use_onecycle")
    ap.add_argument("--onecycle_pct_start", type=float, default=0.4)
    ap.add_argument("--onecycle_div_factor", type=float, default=10.0)
    ap.add_argument("--onecycle_final_div_factor", type=float, default=10000.0)

    ap.add_argument("--freeze_image_backbone", action="store_true", default=True)
    ap.add_argument("--freeze_point_cloud_backbone", action="store_true", default=True)
    ap.add_argument("--unfreeze_point_cloud_backbone", action="store_false", dest="freeze_point_cloud_backbone")

    args = ap.parse_args()

    # 解析 depth_range
    depth_range = tuple(float(x) for x in args.depth_range.split(","))
    voxel_size = tuple(float(x) for x in args.voxel_size.split(","))
    point_cloud_range = tuple(float(x) for x in args.point_cloud_range.split(","))
    code_weights = parse_float_list(args.code_weights, 7, "--code_weights")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------
    # KITTI 路径
    # -------------------------
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
    assert len(ids) > 0, "No KITTI images found."

    # -------------------------
    # build system
    # -------------------------
    point_cloud_feat_index = None if args.point_cloud_feat_index < 0 else args.point_cloud_feat_index

    sys = build_hmfi_system(
        image_cfg=normalize_path(args.image_cfg),
        image_ckpt=normalize_path(args.image_ckpt),
        point_cloud_cfg=normalize_path(args.point_cloud_cfg),
        point_cloud_ckpt=normalize_path(args.point_cloud_ckpt),
        device=args.device,

        image_feat_index=args.image_feat_index,
        point_cloud_feat_index=point_cloud_feat_index,

        lidar_channels=args.lidar_channels,
        image_channels=args.image_channels,
        fusion_out_channels=args.fusion_out_channels,
        d_k=args.d_k,
        d_v=args.d_v,
        num_heads=args.num_heads,
        depth_mode=args.depth_mode,
        downsample_factor=args.downsample_factor,
        depth_range=depth_range,
        num_depth_bins=args.num_depth_bins,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        use_vfim=args.use_vfim,
        vfim_loss_weight=args.vfim_loss_weight,
        alpha=args.alpha,
        pos_iou_thr=args.pos_iou_thr,
        neg_iou_thr=args.neg_iou_thr,
        code_weights=code_weights,
        max_pos_weight=args.max_pos_weight,

        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        pre_nms_topk=args.pre_nms_topk,
        post_nms_topk=args.post_nms_topk,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        debug_base_loss=args.debug_base_loss,

        freeze_image_backbone=args.freeze_image_backbone,
        freeze_point_cloud_backbone=args.freeze_point_cloud_backbone
    )

    if args.mode == "train":
        run_train(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids)
    else:
        if args.nms_sweep:
            base_pred_dir = args.pred_dir
            score_values = [float(x) for x in args.score_thr_sweep.split(",") if x.strip()]
            nms_values = [float(x) for x in args.nms_thr_sweep.split(",") if x.strip()]
            for score_thr in score_values:
                for nms_thr in nms_values:
                    args.score_thr = score_thr
                    args.pred_score_thr = min(args.pred_score_thr, score_thr)
                    args.nms_thr = nms_thr
                    sys.postprocessor.score_thr = score_thr
                    sys.postprocessor.nms_thr = nms_thr
                    args.pred_dir = f"{base_pred_dir}_score{score_thr:g}_nms{nms_thr:g}"
                    print(f"[SWEEP] score_thr={score_thr} nms_thr={nms_thr} pred_dir={args.pred_dir}")
                    run_test(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids)
        else:
            run_test(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids)


if __name__ == "__main__":
    main()
