# -*- coding: utf-8 -*-
"""
test_script.py (LoGoNet KITTI train/test unified version)

功能：
1) mode=test:
   - 在 KITTI val 上调用 LoGoNet 做预测
   - 写出 KITTI label_2 格式预测文件
   - 调用 mmdet3d 的 KITTI eval 输出 bbox / bev / 3d AP

2) mode=train:
   - 使用 build_logonet_system + train_step 做真实训练
   - 读取 KITTI train split
   - 构造 gt_boxes / gt_labels
   - 每轮保存 checkpoint

说明：
- 保留你当前 test_script 的评测逻辑
- 在同一脚本内加入 train 入口
- 训练默认 batch_size=1（当前实现按单样本真实训练，最稳）
"""

from __future__ import annotations

import os
import argparse
import random
from typing import Dict, Any, List, Tuple, Optional

# 激活 fusionrepo 自定义算子注册
import utils  # noqa: F401
import common  # noqa: F401
import fusion_unit  # noqa: F401

import numpy as np
import torch
from PIL import Image
import math
import cv2

from model.logonet.initialization.build_logonet_system import (
    build_logonet_system,
    train_step,
    test_step,
)
from utils.pcdet_kitti_writer import write_kitti_pred_txt_pcdet


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


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    vals = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    return vals


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


def build_lidar_to_image_proj(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """
    KITTI LiDAR point -> image pixel projection.

    P2 alone projects rectified camera coordinates. LoGoNet projects voxel /
    RoI grid points represented in LiDAR coordinates, so the complete matrix is:
        P = P2 @ R0_rect @ Tr_velo_to_cam
    """
    P2 = calib["P2"].astype(np.float32)
    R0 = calib["R0_rect"].astype(np.float32)
    Tr = calib["Tr_velo_to_cam"].astype(np.float32)

    R0_4 = np.eye(4, dtype=np.float32)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4, dtype=np.float32)
    Tr_4[:3, :4] = Tr

    return (P2 @ R0_4 @ Tr_4).astype(np.float32)


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


def adapt_logonet_boxes_for_writer(
    boxes_lidar: torch.Tensor,
    assume_center_z: bool = True
) -> torch.Tensor:
    """
    仅用于 LoGoNet -> KITTI txt 导出前的语义适配。
    当前 writer 内部会对 z 做 +h/2 处理，因此这里可选先把中心 z
    转成底面 z，避免垂直参考点不一致导致 3D AP 异常。
    """
    if boxes_lidar.numel() == 0:
        return boxes_lidar

    boxes_out = boxes_lidar.clone()
    if assume_center_z:
        boxes_out[:, 2] = boxes_out[:, 2] - boxes_out[:, 5] / 2.0
    return boxes_out


# ============================================================
# 3) eval 导入
# ============================================================
def run_kitti_eval(gt_annos, dt_annos, class_names=("Car",)):
    from mmdet3d.evaluation.functional.kitti_utils.eval import kitti_eval  # type: ignore

    dt_annos_safe = []
    for anno in dt_annos:
        anno_safe = dict(anno)
        if "alpha" not in anno_safe or anno_safe["alpha"].shape[0] == 0:
            anno_safe["name"] = np.array([], dtype=object)
            anno_safe["truncated"] = np.array([], dtype=np.float32)
            anno_safe["occluded"] = np.array([], dtype=np.int64)
            anno_safe["alpha"] = np.array([], dtype=np.float32)
            anno_safe["bbox"] = np.zeros((0, 4), dtype=np.float32)
            anno_safe["dimensions"] = np.zeros((0, 3), dtype=np.float32)
            anno_safe["location"] = np.zeros((0, 3), dtype=np.float32)
            anno_safe["rotation_y"] = np.array([], dtype=np.float32)
            anno_safe["score"] = np.array([], dtype=np.float32)
        dt_annos_safe.append(anno_safe)

    current_classes = list(class_names)
    result_str, result_dict = kitti_eval(
        gt_annos=gt_annos,
        dt_annos=dt_annos_safe,
        current_classes=current_classes,
        eval_types=["bbox", "bev", "3d"],
    )
    return {
        "report": result_str,
        "metrics": result_dict
    }


def extract_primary_val_score(res: Dict[str, Any]) -> float:
    metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
    if isinstance(metrics, dict):
        preferred_keys = [
            "KITTI/Car_3D_AP40_moderate_strict",
            "KITTI/Car_3D_AP40_moderate",
            "Car_3D_AP40_moderate_strict",
            "Car_3D_AP40_moderate",
            "KITTI/Car_3D_AP11_moderate_strict",
            "Car_3D_AP11_moderate_strict",
        ]
        for key in preferred_keys:
            val = metrics.get(key, None)
            if isinstance(val, (int, float)):
                return float(val)
        for key, val in metrics.items():
            if isinstance(key, str) and "Car" in key and "3D" in key and "moderate" in key and isinstance(val, (int, float)):
                return float(val)
    return float("-inf")


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
    构造 LoGoNet 单样本测试 batch
    """
    image_tensor, image_meta = load_image_for_mmdet_caffe(img_path, device)
    points_tensor = load_points_tensor(lidar_path, device)

    proj_matrix = torch.from_numpy(build_lidar_to_image_proj(calib)).float().to(device)

    batch = {
        "image": image_tensor,
        "images": image_tensor,
        "point": [points_tensor],
        "points": [points_tensor],
        "image_path": img_path,
        "lidar_path": lidar_path,
        "calib": calib,
        "proj_matrix": proj_matrix,
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
    device: str
) -> Dict[str, Any]:
    """
    构造 LoGoNet 单样本训练 batch
    真实训练所需：
    - images / points
    - gt_boxes (lidar)
    - gt_labels
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

    proj_matrix = torch.from_numpy(build_lidar_to_image_proj(calib)).float().to(device)

    batch = {
        "image": image_tensor,
        "images": image_tensor,
        "point": [points_tensor],
        "points": [points_tensor],
        "image_path": img_path,
        "lidar_path": lidar_path,
        "calib": calib,
        "proj_matrix": proj_matrix,
        "gt_boxes": [gt_boxes],
        "gt_labels": [gt_labels],
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
def _materialize_optional_layers_from_state_dict(sys, state_dict: Dict[str, torch.Tensor]):
    """
    GlobalFusionModule creates channel adapters lazily on first forward.
    Materialize them before load_state_dict so adapter weights from checkpoints
    are restored instead of reported as unexpected_keys.
    """
    gf = getattr(sys.model, "global_fusion_module", None)
    if gf is None:
        return

    lidar_w = state_dict.get("global_fusion_module.lidar_input_adapter.weight")
    if lidar_w is not None and getattr(gf, "lidar_input_adapter", None) is None:
        gf.lidar_input_adapter = torch.nn.Linear(
            in_features=int(lidar_w.shape[1]),
            out_features=int(lidar_w.shape[0]),
            bias=("global_fusion_module.lidar_input_adapter.bias" in state_dict)
        ).to(device=sys.device, dtype=lidar_w.dtype)

    image_w = state_dict.get("global_fusion_module.image_input_adapter.weight")
    if image_w is not None and getattr(gf, "image_input_adapter", None) is None:
        gf.image_input_adapter = torch.nn.Conv2d(
            in_channels=int(image_w.shape[1]),
            out_channels=int(image_w.shape[0]),
            kernel_size=tuple(image_w.shape[2:]),
            stride=1,
            padding=0,
            bias=("global_fusion_module.image_input_adapter.bias" in state_dict)
        ).to(device=sys.device, dtype=image_w.dtype)


def load_fusion_checkpoint(
    sys,
    ckpt_path: str,
    for_test: bool = False,
    restore_optimizer: bool = False
) -> Optional[Dict[str, Any]]:
    ckpt_path = normalize_path(ckpt_path)
    if not ckpt_path:
        if for_test:
            print("[WARN] fusion checkpoint path empty.")
        return None

    if not os.path.exists(ckpt_path):
        if for_test:
            print(f"[WARN] fusion checkpoint not found: {ckpt_path}")
            print("[WARN] 将直接评测当前初始化权重，结果通常不会正常。")
        else:
            print(f"[WARN] resume checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    _materialize_optional_layers_from_state_dict(sys, state_dict)
    missing, unexpected = sys.model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] checkpoint: {ckpt_path}")
    print(f"[LOAD] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        print(f"[LOAD] missing sample={missing[:8]}")
    if unexpected:
        print(f"[LOAD] unexpected sample={unexpected[:8]}")

    if restore_optimizer and (not for_test) and isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            sys.optimizer.load_state_dict(ckpt["optimizer"])
            print("[LOAD] optimizer state restored.")
        except Exception as e:
            print(f"[WARN] optimizer restore failed: {e}")
    if (not for_test) and isinstance(ckpt, dict) and "scheduler" in ckpt and sys.scheduler is not None:
        print("[LOAD] scheduler state is intentionally not restored; a fresh schedule is used for this run.")

    return ckpt if isinstance(ckpt, dict) else {"state_dict": state_dict}


def save_train_checkpoint(sys, ckpt_path: str, epoch: int, metric: Optional[float] = None, metric_name: str = ""):
    ckpt_path = normalize_path(ckpt_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "state_dict": sys.model.state_dict(),
        "optimizer": sys.optimizer.state_dict(),
        "scheduler": sys.scheduler.state_dict() if sys.scheduler is not None else None,
        "metric": None if metric is None else float(metric),
        "metric_name": str(metric_name),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[SAVE] checkpoint saved to: {ckpt_path}")


# ============================================================
# 6) train / test 主流程
# ============================================================
def run_test(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids):
    # -------------------------
    # val split
    # -------------------------
    default_val_split = os.path.join(kitti_root, "ImageSets", "val.txt")
    if args.split_file or os.path.exists(default_val_split):
        split_path = normalize_path(args.split_file) if args.split_file else default_val_split
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
    # 加载训练好的 LoGoNet 权重。训练中调用验证时必须跳过这里，
    # 否则会把当前 epoch 的模型覆盖成磁盘上的旧 checkpoint。
    # -------------------------
    if not getattr(args, "skip_load_ckpt", False):
        test_ckpt = args.fusion_ckpt or args.save_ckpt
        if test_ckpt and (not os.path.exists(normalize_path(test_ckpt))) and os.path.exists(normalize_path(args.save_ckpt)):
            print(f"[WARN] fusion checkpoint not found, fallback to last checkpoint: {test_ckpt}")
            test_ckpt = args.save_ckpt
        load_fusion_checkpoint(sys, test_ckpt, for_test=True)

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

        # 1) 跑 LoGoNet
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

        boxes_for_writer = adapt_logonet_boxes_for_writer(
            boxes_lidar=boxes_lidar,
            assume_center_z=bool(args.writer_fix_logonet_center_z),
        )

        # 2) 写 KITTI txt
        write_kitti_pred_txt_pcdet(
            pred_dir=args.pred_dir,
            frame_id=sid,
            boxes_lidar=boxes_for_writer,
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
    except Exception as e:
        print(f"[WARN] mmdet3d KITTI eval failed: {e}")
        print(f"[DONE] prediction txt saved to: {os.path.join(args.pred_dir, 'data')}")
        return None

    print("\n================ KITTI EVAL REPORT ================\n")
    print(res["report"])
    print("===================================================\n")
    print(f"[DONE] prediction txt saved to: {os.path.join(args.pred_dir, 'data')}")
    return res


def run_train(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids):
    # 使用梯度累积实现 effective batch_size
    # batch_size=4 表示每 4 个样本更新一次参数（与逐样本更新精度等价）

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
    print(f"[TRAIN] batch_size={args.batch_size} (gradient accumulation)")

    # Resume is explicit. Auto-loading stale checkpoints during continued
    # training makes results hard to interpret.
    resume_path = args.resume_ckpt if args.resume_ckpt else ""
    if os.path.exists(resume_path):
        print(f"[RESUME] found checkpoint: {resume_path}")
        load_fusion_checkpoint(sys, resume_path, for_test=False, restore_optimizer=bool(args.resume_optimizer))
    elif resume_path:
        print(f"[WARN] resume checkpoint not found: {resume_path}")
    else:
        print(f"[TRAIN] no checkpoint found, starting from scratch")

    best_loss = 1e18
    accum_steps = args.batch_size

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_ids)

        loss_sum = 0.0
        loss_rpn_sum = 0.0
        loss_conf_sum = 0.0
        loss_reg_sum = 0.0
        step_count = 0

        print(f"\n================ Epoch {epoch}/{args.epochs} ================")

        sys.optimizer.zero_grad(set_to_none=True)
        accum_loss_sum = 0.0
        accum_loss_rpn_sum = 0.0
        accum_loss_conf_sum = 0.0
        accum_loss_reg_sum = 0.0
        accum_count = 0

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
                device=args.device
            )

            # Forward
            sys.model.train()
            if args.freeze_image_backbone:
                sys.model.image_backbone.eval()
            if args.freeze_point_cloud_backbone:
                sys.model.point_cloud_backbone.eval()
            pred_out = sys.model.train_forward(batch)

            from model.logonet.initialization.build_logonet_system import _compute_base_rpn_loss
            base_rpn_loss = _compute_base_rpn_loss(sys, batch)

            loss_dict = sys.criterion(
                pred_out=pred_out,
                batch=batch,
                base_rpn_loss=base_rpn_loss
            )

            loss = loss_dict["loss"] / float(accum_steps)
            loss.backward()
            accum_loss_sum += float(loss_dict["loss"].item())
            accum_loss_rpn_sum += float(loss_dict["loss_rpn"].item())
            accum_loss_conf_sum += float(loss_dict["loss_conf"].item())
            accum_loss_reg_sum += float(loss_dict["loss_reg"].item())
            accum_count += 1

            # 每 accum_steps 个样本更新一次
            if idx % accum_steps == 0:
                # 梯度裁剪
                if sys.grad_clip_norm is not None and sys.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(sys.model.parameters(), max_norm=sys.grad_clip_norm)

                sys.optimizer.step()
                if sys.scheduler is not None:
                    sys.scheduler.step()
                sys.optimizer.zero_grad(set_to_none=True)

                step_count += 1
                cur_loss = accum_loss_sum / max(accum_count, 1)
                cur_loss_rpn = accum_loss_rpn_sum / max(accum_count, 1)
                cur_loss_conf = accum_loss_conf_sum / max(accum_count, 1)
                cur_loss_reg = accum_loss_reg_sum / max(accum_count, 1)
                loss_sum += cur_loss
                loss_rpn_sum += cur_loss_rpn
                loss_conf_sum += cur_loss_conf
                loss_reg_sum += cur_loss_reg
                accum_loss_sum = 0.0
                accum_loss_rpn_sum = 0.0
                accum_loss_conf_sum = 0.0
                accum_loss_reg_sum = 0.0
                accum_count = 0

                if step_count % args.log_interval == 0 or idx == len(train_ids):
                    print(
                        f"[TRAIN] epoch={epoch} step={idx}/{len(train_ids)} "
                        f"loss={cur_loss:.6f} "
                        f"loss_rpn={cur_loss_rpn:.6f} "
                        f"loss_conf={cur_loss_conf:.6f} "
                        f"loss_reg={cur_loss_reg:.6f}"
                    )

        # 处理剩余未更新的梯度
        if len(train_ids) % accum_steps != 0:
            if sys.grad_clip_norm is not None and sys.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(sys.model.parameters(), max_norm=sys.grad_clip_norm)
            sys.optimizer.step()
            if sys.scheduler is not None:
                sys.scheduler.step()
            sys.optimizer.zero_grad(set_to_none=True)
            step_count += 1
            cur_loss = accum_loss_sum / max(accum_count, 1)
            cur_loss_rpn = accum_loss_rpn_sum / max(accum_count, 1)
            cur_loss_conf = accum_loss_conf_sum / max(accum_count, 1)
            cur_loss_reg = accum_loss_reg_sum / max(accum_count, 1)
            loss_sum += cur_loss
            loss_rpn_sum += cur_loss_rpn
            loss_conf_sum += cur_loss_conf
            loss_reg_sum += cur_loss_reg

        mean_loss = loss_sum / max(step_count, 1)
        mean_rpn = loss_rpn_sum / max(step_count, 1)
        mean_conf = loss_conf_sum / max(step_count, 1)
        mean_reg = loss_reg_sum / max(step_count, 1)

        print(
            f"[EPOCH END] epoch={epoch} "
            f"mean_loss={mean_loss:.6f} "
            f"mean_loss_rpn={mean_rpn:.6f} "
            f"mean_loss_conf={mean_conf:.6f} "
            f"mean_loss_reg={mean_reg:.6f}"
        )

        # 保存 last
        save_train_checkpoint(sys, args.save_ckpt, epoch)

        current_lr = sys.optimizer.param_groups[0]["lr"]
        print(f"[LR] epoch={epoch} lr={current_lr:.6f}")

        # Training loss is not a reliable KITTI AP selector. Keep it separately
        # for debugging; AP selection remains a manual test-mode step.
        best_loss_ckpt = args.save_ckpt.replace(".pth", "_best_loss.pth")
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_train_checkpoint(sys, best_loss_ckpt, epoch, metric=mean_loss, metric_name="train_loss")

        # KITTI official evaluation is kept as an explicit test-mode step.
        # Training only saves last and best-loss checkpoints.


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
    ap.add_argument("--pred_dir", type=str, default="./pred_kitti")
    ap.add_argument("--pred_score_thr", type=float, default=0.0)
    ap.add_argument(
        "--writer_fix_logonet_center_z",
        type=int,
        default=1,
        help="1: export前将LoGoNet的中心z转为底面z再交给writer；0: 关闭该适配",
    )

    # -------------------------
    # 训练参数
    # -------------------------
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_ckpt", type=str, default="./checkpoints/logonet_fusion_trained.pth")
    ap.add_argument("--resume_ckpt", type=str, default="")
    ap.add_argument("--resume_optimizer", action="store_true", default=False)
    ap.add_argument("--val_every", type=int, default=0, help="deprecated; training does not run eval automatically")

    # -------------------------
    # 2D / 3D 后端配置
    # -------------------------
    ap.add_argument(
        "--image_cfg",
        type=str,
        default="/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet/.mim/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py"
    )
    ap.add_argument(
        "--point_cloud_cfg",
        type=str,
        default="/home/lzy/miniconda3/envs/epnet/lib/python3.8/site-packages/mmdet3d/.mim/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py"
    )
    ap.add_argument("--image_ckpt", type=str, default="./checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth")
    ap.add_argument("--point_cloud_ckpt", type=str, default="./checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth")
    ap.add_argument("--fusion_ckpt", type=str, default="./checkpoints/logonet_fusion_trained.pth")

    # -------------------------
    # build_logonet_system 参数
    # -------------------------
    ap.add_argument("--image_feat_index", type=int, default=0)
    ap.add_argument("--point_cloud_feat_index", type=int, default=-1)
    ap.add_argument("--prediction_branch", type=str, default="pts_bbox")
    ap.add_argument("--use_voxelize", action="store_true", default=True)
    ap.add_argument(
        "--point_cloud_allowed_labels",
        type=str,
        default="2",
        help="逗号分隔的3D proposal类别过滤列表；空字符串表示不过滤",
    )

    ap.add_argument("--lidar_channels", type=int, default=128)
    ap.add_argument("--image_channels", type=int, default=256)
    ap.add_argument("--pos_channels", type=int, default=128)
    ap.add_argument("--roi_grid_size", type=int, default=6)
    ap.add_argument("--roi_pool_mode", type=str, default="mean")
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--ffn_ratio", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=1)
    ap.add_argument("--box_dim", type=int, default=7)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--pos_iou_thr", type=float, default=0.55)
    ap.add_argument("--neg_iou_thr", type=float, default=0.25)
    ap.add_argument("--roi_per_image", type=int, default=128)
    ap.add_argument("--fg_ratio", type=float, default=0.5)
    ap.add_argument("--cls_fg_thresh", type=float, default=0.75)
    ap.add_argument("--cls_bg_thresh", type=float, default=0.25)
    ap.add_argument("--cls_bg_thresh_lo", type=float, default=0.1)
    ap.add_argument("--reg_fg_thresh", type=float, default=0.55)
    ap.add_argument("--hard_bg_ratio", type=float, default=0.8)
    ap.add_argument("--cls_score_type", type=str, default="roi_iou")
    ap.add_argument("--score_thr", type=float, default=0.0)
    ap.add_argument("--nms_thr", type=float, default=0.5)
    ap.add_argument("--pre_nms_topk", type=int, default=300)
    ap.add_argument("--post_nms_topk", type=int, default=100)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip_norm", type=float, default=10.0)

    ap.add_argument("--freeze_image_backbone", action="store_true", default=True)
    ap.add_argument("--freeze_point_cloud_backbone", action="store_true", default=True)
    ap.add_argument("--unfreeze_point_cloud_backbone", action="store_false", dest="freeze_point_cloud_backbone")
    ap.add_argument("--use_roi_score_fusion_in_loss", action="store_true", default=False)
    ap.add_argument("--fuse_roi_score_in_test", action="store_true", default=True)

    args = ap.parse_args()

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
    allowed_labels = parse_int_list(args.point_cloud_allowed_labels)
    if len(allowed_labels) == 0:
        allowed_labels = None

    sys = build_logonet_system(
        image_cfg=normalize_path(args.image_cfg),
        image_ckpt=normalize_path(args.image_ckpt),
        point_cloud_cfg=normalize_path(args.point_cloud_cfg),
        point_cloud_ckpt=normalize_path(args.point_cloud_ckpt),
        device=args.device,

        image_feat_index=args.image_feat_index,
        point_cloud_feat_index=point_cloud_feat_index,
        prediction_branch=args.prediction_branch,
        use_voxelize=args.use_voxelize,
        point_cloud_allowed_labels=allowed_labels,

        lidar_channels=args.lidar_channels,
        image_channels=args.image_channels,
        pos_channels=args.pos_channels,
        roi_grid_size=args.roi_grid_size,
        roi_pool_mode=args.roi_pool_mode,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        num_classes=args.num_classes,
        box_dim=args.box_dim,

        alpha=args.alpha,
        pos_iou_thr=args.pos_iou_thr,
        neg_iou_thr=args.neg_iou_thr,
        roi_per_image=args.roi_per_image,
        fg_ratio=args.fg_ratio,
        cls_fg_thresh=args.cls_fg_thresh,
        cls_bg_thresh=args.cls_bg_thresh,
        cls_bg_thresh_lo=args.cls_bg_thresh_lo,
        reg_fg_thresh=args.reg_fg_thresh,
        hard_bg_ratio=args.hard_bg_ratio,
        cls_score_type=args.cls_score_type,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        pre_nms_topk=args.pre_nms_topk,
        post_nms_topk=args.post_nms_topk,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,

        freeze_image_backbone=args.freeze_image_backbone,
        freeze_point_cloud_backbone=args.freeze_point_cloud_backbone,
        use_roi_score_fusion_in_loss=args.use_roi_score_fusion_in_loss,
        fuse_roi_score_in_test=args.fuse_roi_score_in_test,
        epochs=args.epochs,
        steps_per_epoch=1500,
    )

    if args.mode == "train":
        run_train(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids)
    else:
        run_test(args, sys, image_dir, lidar_dir, calib_dir, label_dir, kitti_root, ids)


if __name__ == "__main__":
    main()
