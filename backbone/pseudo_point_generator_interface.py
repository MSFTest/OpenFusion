# -*- coding: utf-8 -*-
"""
PseudoPointGeneratorInterface

伪点云生成接口 - 使用MiDaS预训练深度估计模型

基于MiDaS (Intel)的预训练深度估计模型，从单目图像估计深度图，
然后结合相机内外参投影到3D空间生成伪点云。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import numpy as np


class PseudoPointGeneratorInterface(nn.Module):
    """
    伪点云生成接口 - 使用MiDaS预训练深度估计

    使用MiDaS预训练模型从图像估计深度图，
    然后通过相机内外参投影到3D空间生成伪点云。

    输入:
        images: (B, 3, H, W) RGB图像，像素范围为[0,1]或[0,255]
        proj_matrix: (B, 4, 4) 投影矩阵
        lidar_to_cam: (B, 4, 4) LiDAR到相机坐标变换
        cam_intrinsic: (B, 3, 3) 相机内参

    输出:
        dict with keys:
            - pseudo_points: (N, 3+C) 伪点云坐标+特征
            - pseudo_coords: (N, 3) 伪点云坐标
            - pseudo_features: (N, C) 伪点云特征
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        feat_dim: int = 256,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        device: str = "cuda:0",
        midas_checkpoint: Optional[str] = None,
        allow_depth_fallback: bool = False,
        enable_legacy_img_features: bool = True,
    ):
        """
        Args:
            model_type: MiDaS模型类型；非DPT_Large须显式提供匹配权重
            feat_dim: 输出特征维度
            depth_min: 最小深度阈值
            depth_max: 最大深度阈值
            device: 模型设备
            midas_checkpoint: 可选的本地MiDaS权重路径
            allow_depth_fallback: 显式允许MiDaS加载失败后使用常量深度
            enable_legacy_img_features: 创建旧版RGB/MiDaS投影层供直接调用者兼容

        The cached MiDaS repository and checkpoint are trusted local inputs.
        Deployment must verify the checkpoint SHA before loading it.
        """
        super().__init__()
        self.model_type = model_type
        self.feat_dim = feat_dim
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.device = device
        self.enable_legacy_img_features = bool(enable_legacy_img_features)

        # 加载预训练MiDaS模型
        self.using_depth_fallback = False
        self.depth_fallback_reason = None
        self._load_midas_model(model_type, device, midas_checkpoint, allow_depth_fallback)

        if self.enable_legacy_img_features:
            self.img_feat_map = nn.Sequential(
                nn.Conv2d(3, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, feat_dim, 1)
            )
            self.feat_map = nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, feat_dim, 1)
            )

    def _load_midas_model(
        self,
        model_type: str,
        device: str,
        midas_checkpoint: Optional[str],
        allow_depth_fallback: bool
    ):
        """Load MiDaS only from the local torch hub cache."""
        if model_type != "DPT_Large" and not midas_checkpoint:
            raise ValueError(
                f"model_type={model_type!r} requires an explicit matching MiDaS checkpoint"
            )

        try:
            hub_dir = Path(torch.hub.get_dir())
            repos = [hub_dir / "intel-isl_MiDaS_master", hub_dir / "intel-isl_MiDaS_main"]
            repo = next(
                (candidate for candidate in repos if candidate.is_dir() and (candidate / "hubconf.py").is_file()),
                None,
            )
            if repo is None:
                raise FileNotFoundError(
                    "MiDaS repository cache with hubconf.py not found; expected one of: "
                    + ", ".join(str(path) for path in repos)
                )

            checkpoint_path = Path(midas_checkpoint) if midas_checkpoint else hub_dir / "checkpoints" / "dpt_large_384.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"MiDaS checkpoint not found: {checkpoint_path}")

            self.midas = torch.hub.load(
                str(repo), model_type, source="local", pretrained=False, verbose=False
            )
            state_dict = torch.load(str(checkpoint_path), map_location=device)
            self.midas.load_state_dict(state_dict)
            self.midas.to(device)
            self.midas.eval()
            midas_transforms = torch.hub.load(
                str(repo), "transforms", source="local", verbose=False
            )
            self.midas_transform = midas_transforms.dpt_transform
            self.midas_checkpoint = checkpoint_path
            print(
                "[PseudoPointGenerator] Loaded trusted local MiDaS: "
                f"{repo} ({checkpoint_path}); verify checkpoint SHA during deployment"
            )
        except Exception as e:
            if not allow_depth_fallback:
                raise RuntimeError(f"Failed to load local MiDaS model: {e}") from e
            self.midas = None
            self.midas_transform = None
            self.midas_checkpoint = None
            self.using_depth_fallback = True
            self.depth_fallback_reason = str(e)
            print(f"[WARNING] Explicit MiDaS depth fallback enabled: {e}")

    def _estimate_depth_midas(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用MiDaS估计深度图

        Args:
            images: (B, 3, H, W) 像素范围为[0,1]或[0,255]的RGB图像

        Returns:
            depth: (B, 1, H, W) 深度图
        """
        B, C, H, W = images.shape

        if self.midas is not None:
            if not torch.isfinite(images).all():
                raise ValueError("MiDaS images must be finite and within 0..255")
            image_min = float(images.min())
            image_max = float(images.max())
            tolerance = 1e-4
            if image_min < -tolerance or image_max > 255.0 + tolerance:
                raise ValueError("MiDaS images must be finite and within 0..255")
            midas_images = images.float().clamp(0.0, 255.0)
            if image_max <= 1.0 + tolerance:
                midas_images = midas_images * 255.0

            with torch.no_grad():
                # 使用MiDaS的transform进行预处理
                if self.midas_transform is not None:
                    # MiDaS的transform期望单张图像 (H, W, 3)
                    # 需要将batch拆开处理
                    depth_list = []
                    for b in range(B):
                        img_b = midas_images[b]  # (3, H, W), 0..255 as expected by MiDaS transforms
                        # 转为 (H, W, 3)
                        img_bhw3 = img_b.permute(1, 2, 0).cpu().numpy()
                        # 应用transform
                        input_batch = self.midas_transform(img_bhw3).to(images.device)
                        # 估计深度
                        depth_b = self.midas(input_batch)
                        if depth_b.ndim == 4:
                            if depth_b.shape[0] != 1:
                                raise ValueError("MiDaS must return one depth map per transformed image")
                            depth_b = depth_b.squeeze(0)
                        elif depth_b.ndim == 2:
                            depth_b = depth_b.unsqueeze(0)
                        if depth_b.ndim != 3:
                            raise ValueError(f"Unexpected MiDaS depth output shape: {tuple(depth_b.shape)}")
                        depth_list.append(depth_b)

                    # 堆叠并调整回原始尺寸
                    depth = torch.stack(depth_list, dim=0)  # (B, 1, H', W')
                else:
                    # Fallback: 简单的resize
                    depth = F.interpolate(
                        images.mean(dim=1, keepdim=True),
                        size=(384, 384),
                        mode='bilinear',
                        align_corners=False
                    )
                    depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)

                # 确保输出是 (B, 1, H, W)
                if depth.dim() == 3:
                    depth = depth.unsqueeze(1)

                return depth
        else:
            # Fallback: 使用简单的基于平面的假设
            depth = torch.ones((B, 1, H, W), device=images.device) * 30.0  # 假设所有点在30米处
            return depth

    def _project_to_3d(
        self,
        images: torch.Tensor,
        depth: torch.Tensor,
        proj_matrix: Optional[torch.Tensor] = None,
        lidar_to_cam: Optional[torch.Tensor] = None,
        cam_intrinsic: Optional[torch.Tensor] = None,
        ori_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        resize_shapes: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        将深度图投影到3D点云

        Args:
            images: (B, 3, H, W) 图像
            depth: (B, 1, H, W) 深度图
            proj_matrix: (B, 4, 4) 投影矩阵
            lidar_to_cam: (B, 4, 4) LiDAR到相机变换
            cam_intrinsic: (B, 3, 3) 相机内参

        Returns:
            coords: (B, H, W, 3) 3D坐标
        """
        B, C, H, W = images.shape

        # 生成像素网格
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=depth.dtype),
            torch.arange(W, device=depth.device, dtype=depth.dtype),
            indexing='ij'
        )  # (H, W), (H, W)
        resize_uv = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        if ori_shapes is not None and resize_shapes is not None:
            pixel_uv = self._resize_uv_to_original(resize_uv, ori_shapes, resize_shapes)
        else:
            pixel_uv = resize_uv

        p2 = None
        if proj_matrix is not None:
            if proj_matrix.dim() == 2:
                proj_matrix = proj_matrix.unsqueeze(0)
            if proj_matrix.dim() != 3 or proj_matrix.shape[-2] < 3 or proj_matrix.shape[-1] < 4:
                raise ValueError("proj_matrix must contain a 3x4 P2 matrix")
            if proj_matrix.shape[0] == 1 and B > 1:
                proj_matrix = proj_matrix.expand(B, -1, -1)
            if proj_matrix.shape[0] != B:
                raise ValueError("proj_matrix must have one matrix per image")
            p2 = proj_matrix.to(device=depth.device, dtype=depth.dtype)[:, :3, :4]

        if cam_intrinsic is not None:
            if cam_intrinsic.dim() == 2:
                cam_intrinsic = cam_intrinsic.unsqueeze(0)
            if cam_intrinsic.dim() == 3 and cam_intrinsic.shape[0] == 1 and B > 1:
                cam_intrinsic = cam_intrinsic.expand(B, -1, -1)
            if cam_intrinsic.shape[0] != B:
                raise ValueError("cam_intrinsic must have one matrix per image")
            cam_intrinsic = cam_intrinsic.to(device=depth.device, dtype=depth.dtype)
            if p2 is not None and not torch.allclose(cam_intrinsic, p2[:, :, :3], rtol=1e-5, atol=1e-5):
                raise ValueError("cam_intrinsic conflicts with the left 3x3 block of P2")
        elif p2 is not None:
            cam_intrinsic = p2[:, :, :3]

        if cam_intrinsic is None:
            # Legacy default KITTI intrinsic when neither K nor P2 is supplied.
            fx = torch.full((B,), 721.5377, device=depth.device, dtype=depth.dtype)
            fy = torch.full((B,), 721.5377, device=depth.device, dtype=depth.dtype)
            cx = torch.full((B,), 609.5593, device=depth.device, dtype=depth.dtype)
            cy = torch.full((B,), 172.854, device=depth.device, dtype=depth.dtype)
        else:
            fx = cam_intrinsic[:, 0, 0]
            fy = cam_intrinsic[:, 1, 1]
            cx = cam_intrinsic[:, 0, 2]
            cy = cam_intrinsic[:, 1, 2]
        tx = p2[:, 0, 3] if p2 is not None else torch.zeros_like(fx)
        ty = p2[:, 1, 3] if p2 is not None else torch.zeros_like(fy)

        # 展开深度
        depth = depth.squeeze(1)  # (B, H, W)

        # 计算3D坐标（相机坐标系下）
        # Invert P2's first two projected coordinates, including its translation column.
        # z = depth
        x = (pixel_uv[..., 0] * depth - cx.view(B, 1, 1) * depth - tx.view(B, 1, 1)) / fx.view(B, 1, 1)
        y = (pixel_uv[..., 1] * depth - cy.view(B, 1, 1) * depth - ty.view(B, 1, 1)) / fy.view(B, 1, 1)
        z = depth

        coords_cam = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)

        # 如果有lidar_to_cam变换，转换到LiDAR坐标系
        if lidar_to_cam is not None:
            # 处理 lidar_to_cam 形状问题
            if lidar_to_cam.dim() == 2:
                # 如果是 (4, 4)，扩展为 (1, 4, 4)
                lidar_to_cam = lidar_to_cam.unsqueeze(0)
            if lidar_to_cam.dim() == 3 and lidar_to_cam.shape[0] == 1 and B > 1:
                # 如果是 (1, 4, 4) 但 B > 1，扩展到 (B, 4, 4)
                lidar_to_cam = lidar_to_cam.expand(B, -1, -1)
            lidar_to_cam = lidar_to_cam.to(device=coords_cam.device, dtype=coords_cam.dtype)

            # coords_lidar = T @ coords_cam (T = cam_to_lidar)
            # 先求逆得到 cam_to_lidar
            cam_to_lidar = torch.inverse(lidar_to_cam)  # (B, 4, 4)

            # 扩展到齐次坐标
            ones = torch.ones((*coords_cam.shape[:3], 1), device=coords_cam.device, dtype=coords_cam.dtype)
            coords_homo = torch.cat([coords_cam, ones], dim=-1)  # (B, H, W, 4)

            # 变换
            coords_lidar = torch.einsum('bij,bhwj->bhwi', cam_to_lidar[:, :3, :], coords_homo)

            return coords_lidar
        else:
            return coords_cam

    @staticmethod
    def _shape_batch_tensor(shapes, batch_size: int, device, dtype, name: str) -> torch.Tensor:
        shape_tensor = torch.as_tensor(shapes, device=device, dtype=dtype)
        if shape_tensor.ndim == 1 and shape_tensor.numel() == 2:
            shape_tensor = shape_tensor.unsqueeze(0)
        if shape_tensor.shape != (batch_size, 2):
            raise ValueError(f"{name} must contain one (height, width) pair per image")
        return shape_tensor

    @staticmethod
    def _resize_uv_to_original(
        resize_uv: torch.Tensor,
        ori_shapes: Sequence[Tuple[int, int]],
        resize_shapes: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        """Map resize-grid pixel centers back to the original image coordinates."""
        if resize_uv.ndim != 4 or resize_uv.shape[-1] != 2:
            raise ValueError("resize_uv must have shape (B, H, W, 2)")
        shape_device = resize_uv.device
        ori = PseudoPointGeneratorInterface._shape_batch_tensor(
            ori_shapes, resize_uv.shape[0], shape_device, resize_uv.dtype, "ori_shapes"
        )
        resized = PseudoPointGeneratorInterface._shape_batch_tensor(
            resize_shapes, resize_uv.shape[0], shape_device, resize_uv.dtype, "resize_shapes"
        )
        if (ori <= 0).any() or (resized <= 0).any():
            raise ValueError("image dimensions must be positive")
        scale = torch.stack([ori[:, 1] / resized[:, 1], ori[:, 0] / resized[:, 0]], dim=-1)
        return (resize_uv + 0.5) * scale[:, None, None, :] - 0.5

    def forward(
        self,
        images: torch.Tensor,
        proj_matrix: Optional[torch.Tensor] = None,
        lidar_to_cam: Optional[torch.Tensor] = None,
        cam_intrinsic: Optional[torch.Tensor] = None,
        ori_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        resize_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        compute_legacy_img_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) 原始图像 [0, 1] 归一化
            proj_matrix: (B, 4, 4) 投影矩阵
            lidar_to_cam: (B, 4, 4) LiDAR到相机变换
            cam_intrinsic: (B, 3, 3) 相机内参

        Returns:
            dict with pseudo_points, pseudo_coords, pseudo_features, flat_pseudo_point_batch_indices
        """
        if compute_legacy_img_features and not getattr(self, "enable_legacy_img_features", True):
            raise ValueError(
                "compute_legacy_img_features=True requires enable_legacy_img_features=True"
            )
        B, C, H, W = images.shape
        if (ori_shapes is None) != (resize_shapes is None):
            raise ValueError("ori_shapes and resize_shapes must be provided together")
        if ori_shapes is not None:
            ori_shapes = self._shape_batch_tensor(ori_shapes, B, images.device, torch.float32, "ori_shapes")
            resize_shapes = self._shape_batch_tensor(resize_shapes, B, images.device, torch.float32, "resize_shapes")
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=images.device, dtype=torch.float32),
            torch.arange(W, device=images.device, dtype=torch.float32),
            indexing='ij',
        )

        # 1. 使用MiDaS估计深度图
        depth = self._estimate_depth_midas(images)  # (B, 1, H', W') MiDaS输出分辨率

        # 确保depth和images尺寸一致（MiDaS输出可能与原图不同）
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(
                depth,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # (B, 1, H, W)

        # 2. 投影到3D坐标
        coords_3d = self._project_to_3d(
            images, depth,
            proj_matrix=proj_matrix,
            lidar_to_cam=lidar_to_cam,
            cam_intrinsic=cam_intrinsic,
            ori_shapes=ori_shapes,
            resize_shapes=resize_shapes,
        )  # (B, H, W, 3)

        # Direct callers retain the historical RGB feature map; MPCF samples its FPN instead.
        if compute_legacy_img_features:
            img_features = self.img_feat_map(images)
            img_features = img_features.permute(0, 2, 3, 1).reshape(B, H * W, self.feat_dim)
        else:
            img_features = images.new_zeros((B, H * W, self.feat_dim))

        # 4. 过滤有效深度范围内的点
        depth_flat = depth.squeeze(1).view(B, -1)  # (B, H*W)
        valid_mask = (depth_flat >= self.depth_min) & (depth_flat <= self.depth_max)  # (B, H*W)
        if ori_shapes is not None and resize_shapes is not None:
            height_width = ori_shapes.to(device=images.device, dtype=coords_3d.dtype)
            mapped_uv = self._resize_uv_to_original(
                torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1),
                ori_shapes,
                resize_shapes,
            )
            valid_uv = (
                (mapped_uv[..., 0] >= -0.5)
                & (mapped_uv[..., 0] < height_width[:, None, None, 1] - 0.5)
                & (mapped_uv[..., 1] >= -0.5)
                & (mapped_uv[..., 1] < height_width[:, None, None, 0] - 0.5)
            )
            valid_mask &= valid_uv.reshape(B, -1)
        else:
            mapped_uv = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # 5. 展平
        coords_flat = coords_3d.view(B, -1, 3)  # (B, H*W, 3)
        feat_flat = img_features  # (B, H*W, feat_dim)

        # 6. 组合伪点云
        pseudo_points_list = []
        pseudo_coords_list = []
        pseudo_features_list = []
        pseudo_batch_indices_list = []
        pseudo_uv_list = []
        pseudo_pixel_indices_list = []

        for b in range(B):
            mask = valid_mask[b]
            n_valid = mask.sum().item()

            if n_valid == 0:
                continue

            coords_b = coords_flat[b, mask]  # (N, 3)
            feat_b = feat_flat[b, mask]  # (N, feat_dim)

            # 拼接坐标和特征
            pseudo_point_b = torch.cat([coords_b, feat_b], dim=-1)  # (N, 3+feat_dim)

            pseudo_points_list.append(pseudo_point_b)
            pseudo_batch_indices_list.append(torch.full((n_valid,), b, device=images.device, dtype=torch.long))
            pseudo_coords_list.append(coords_b)
            pseudo_features_list.append(feat_b)
            pseudo_uv_list.append(mapped_uv[b].reshape(-1, 2)[mask])
            pseudo_pixel_indices_list.append(torch.arange(H * W, device=images.device)[mask])

        if len(pseudo_points_list) == 0:
            device = images.device
            return {
                "pseudo_points": torch.zeros((0, 3 + self.feat_dim), device=device, dtype=torch.float32),
                "pseudo_coords": torch.zeros((0, 3), device=device, dtype=torch.float32),
                "pseudo_features": torch.zeros((0, self.feat_dim), device=device, dtype=torch.float32),
                "flat_pseudo_point_batch_indices": torch.zeros((0,), device=device, dtype=torch.long),
                "pseudo_uv": torch.zeros((0, 2), device=device, dtype=torch.float32),
                "pseudo_pixel_indices": torch.zeros((0,), device=device, dtype=torch.long),
            }

        # 拼接所有batch的点
        pseudo_points = torch.cat(pseudo_points_list, dim=0)  # (N, 3+feat_dim)
        pseudo_coords = torch.cat(pseudo_coords_list, dim=0)  # (N, 3)
        pseudo_features = torch.cat(pseudo_features_list, dim=0)  # (N, feat_dim)

        return {
            "pseudo_points": pseudo_points,
            "pseudo_coords": pseudo_coords,
            "pseudo_features": pseudo_features,
            "flat_pseudo_point_batch_indices": torch.cat(pseudo_batch_indices_list, dim=0),
            "pseudo_uv": torch.cat(pseudo_uv_list, dim=0),
            "pseudo_pixel_indices": torch.cat(pseudo_pixel_indices_list, dim=0),
        }
