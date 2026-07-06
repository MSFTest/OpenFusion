# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from pcdet.utils.calibration_kitti import Calibration
from pcdet.utils.box_utils import (
    boxes3d_lidar_to_kitti_camera,
    boxes3d_kitti_camera_to_imageboxes
)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def project_box_to_image(box_cam, calib):
    import numpy as np

    x, y, z, l, w, h, ry = box_cam

    corners = np.array([
        [ l/2,  w/2, 0],
        [ l/2, -w/2, 0],
        [-l/2, -w/2, 0],
        [-l/2,  w/2, 0],
        [ l/2,  w/2, -h],
        [ l/2, -w/2, -h],
        [-l/2, -w/2, -h],
        [-l/2,  w/2, -h],
    ])

    R = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    corners = corners @ R.T
    corners += np.array([x, y, z])

    pts_img = calib.project_rect_to_image(corners)

    x1, y1 = pts_img.min(axis=0)
    x2, y2 = pts_img.max(axis=0)

    return [x1, y1, x2, y2]

class CalibAdapter:
    """
    适配 OpenPCDet box_utils 所需的 calibration 接口
    输入 calib 为 dict:
        {
            "P2": (3,4),
            "R0_rect": (3,3),
            "Tr_velo_to_cam": (3,4)
        }
    """

    def __init__(self, calib: dict):
        self.P2 = _to_numpy(calib["P2"]).astype(np.float32)
        self.R0 = _to_numpy(calib["R0_rect"]).astype(np.float32)
        self.V2C = _to_numpy(calib["Tr_velo_to_cam"]).astype(np.float32)

        self.R0_ext = np.eye(4, dtype=np.float32)
        self.R0_ext[:3, :3] = self.R0

        self.V2C_ext = np.eye(4, dtype=np.float32)
        self.V2C_ext[:3, :4] = self.V2C

    def lidar_to_rect(self, pts_lidar: np.ndarray) -> np.ndarray:
        pts_lidar = np.asarray(pts_lidar, dtype=np.float32)
        pts_hom = np.concatenate(
            [pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)],
            axis=1
        )
        pts_rect = (self.R0_ext @ (self.V2C_ext @ pts_hom.T)).T
        return pts_rect[:, :3]

    def rect_to_img(self, pts_rect: np.ndarray):
        pts_rect = np.asarray(pts_rect, dtype=np.float32)
        pts_hom = np.concatenate(
            [pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)],
            axis=1
        )
        pts_2d_hom = (self.P2 @ pts_hom.T).T
        pts_img = pts_2d_hom[:, 0:2] / np.clip(pts_2d_hom[:, 2:3], a_min=1e-6, a_max=None)
        pts_depth = pts_2d_hom[:, 2]
        return pts_img, pts_depth

    def corners3d_to_img_boxes(self, corners3d: np.ndarray):
        """
        OpenPCDet 的 boxes3d_kitti_camera_to_imageboxes 会调用这个接口
        corners3d: (N,8,3)
        """
        sample_num = corners3d.shape[0]
        corners_hom = np.concatenate(
            [corners3d, np.ones((sample_num, 8, 1), dtype=np.float32)],
            axis=2
        )  # (N,8,4)

        img_pts = corners_hom @ self.P2.T  # (N,8,3)
        x = img_pts[:, :, 0] / np.clip(img_pts[:, :, 2], a_min=1e-6, a_max=None)
        y = img_pts[:, :, 1] / np.clip(img_pts[:, :, 2], a_min=1e-6, a_max=None)

        x1 = np.min(x, axis=1)
        y1 = np.min(y, axis=1)
        x2 = np.max(x, axis=1)
        y2 = np.max(y, axis=1)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        boxes_corner = np.stack([x, y], axis=2).astype(np.float32)
        return boxes, boxes_corner


def write_kitti_pred_txt_pcdet(
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
    直接使用 OpenPCDet 的 KITTI 转换逻辑
    约定输入 boxes_lidar 当前来自你的工程:
        [x, y, z, w, l, h, yaw]
    OpenPCDet 期望:
        [x, y, z, l, w, h, yaw]
    所以这里会自动交换第4/5维
    """
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
    boxes_pcdet[:, 2] = boxes_pcdet[:, 2] + boxes_pcdet[:, 5] / 2.0
    calib_obj = Calibration(calib_path)

    # OpenPCDet 逻辑：LiDAR -> KITTI camera
    boxes_camera = boxes3d_lidar_to_kitti_camera(boxes_pcdet, calib_obj)

    # OpenPCDet 逻辑：camera box -> image box
    boxes_img = boxes3d_kitti_camera_to_imageboxes(
        boxes_camera,
        calib_obj,
        image_shape=image_shape
    )
    # OpenPCDet 逻辑里的 alpha
    alpha = boxes_camera[:, 6] - np.arctan2(boxes_camera[:, 0], boxes_camera[:, 2])

    lines = []
    H, W = int(image_shape[0]), int(image_shape[1])

    for i in range(boxes_camera.shape[0]):
        x1, y1, x2, y2 = boxes_img[i]
        x1 = np.clip(x1, 0, W - 1)
        x2 = np.clip(x2, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        y2 = np.clip(y2, 0, H - 1)

        x, y, z = boxes_camera[i, 0:3]
        l, h, w = boxes_camera[i, 3:6]
        ry = boxes_camera[i, 6]
        score = scores[i]
        # 跳过无效框
        if z <= 0:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # 关键修正：OpenPCDet 这里的 y 是 box center，
        # KITTI label_2 需要的是 bottom center

        line = (
            f"{cls_name} 0 0 {alpha[i]:.6f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.6f} {w:.6f} {l:.6f} "
            f"{x:.6f} {y:.6f} {z:.6f} "
            f"{ry:.6f} {score:.6f}"
        )
        lines.append(line)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path