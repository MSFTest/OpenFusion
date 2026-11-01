"""F03：解码 query box、投影相机并采样图像 ROI。"""

from __future__ import annotations

import torch
from torchvision.ops import roi_align

from utils.deepinteraction_decode_boxes import deepinteraction_decode_boxes as decode_boxes
from utils.deepinteraction_lidar_box_corners import deepinteraction_lidar_box_corners as lidar_box_corners


def _impl(image_feature, center, height, dimension, rotation, velocity,
          lidar2image, lidar_augmentation, image_size, image_shape,
          image_crop_offset, image_flip, point_cloud_range,
          voxel_size, out_size_factor: int, output_size: int,
          sampling_ratio: int):
    """返回 ROI、每相机可见 mask 和解码框。

    Args:
        image_feature: ``[B,V,C,Hi,Wi]`` stride-4 图像特征。
        center/height/dimension/rotation/velocity: 原预测字段 ``[B,D,K]``。
        image_size: ``[B,2]``，原 ``input_shape``/pad shape。
        image_shape: ``[B,2]``，水平翻转前的 ``(H,W)``。
        image_crop_offset: ``[B,2]``，scale 后裁剪的 ``(x,y)`` 偏移。
        image_flip: ``[B]`` bool，是否执行原水平翻转。
    Returns:
        ``roi[B,V,K,C,S,S]``、``visible[B,V,K]``、``boxes[B,K,9]``。
        ROIAlign 仅对图像特征可微；预测字段在 decode 前按原源码 detach，
        因而 box 坐标与可见性判断均不可微。

    保留原框中心可见性、八角点外接矩形、ROIAlignV2(output=7, scale=1/4,
    sampling_ratio=2) 及后相机覆盖所需的逐视图布局。原实现只有当前视图
    可见 query 数量大于 1 时才处理，本 reference 同样保留此边界。
    原源码：``decoder_utils.py:632-761``，``ImageRCNNBlock.forward``；
    ``transfusion_bbox_coder.py:39-126``。
    """
    boxes = decode_boxes(
        center.detach(), height.detach(), dimension.detach(), rotation.detach(),
        velocity.detach(),
        point_cloud_range, voxel_size, out_size_factor
    )
    corners = lidar_box_corners(boxes)
    batch, views, channels = image_feature.shape[:3]
    proposals = center.shape[-1]
    pooled = image_feature.new_zeros(
        (batch, views, proposals, channels, output_size, output_size)
    )
    visible = torch.zeros(
        (batch, views, proposals), dtype=torch.bool, device=image_feature.device
    )
    point_cloud_range = point_cloud_range.to(
        device=center.device, dtype=center.dtype
    )
    voxel_size = voxel_size.to(device=center.device, dtype=center.dtype)
    centers = torch.cat(
        (center.permute(0, 2, 1) * out_size_factor * voxel_size[0] + point_cloud_range[0],
         height.permute(0, 2, 1)), -1
    )
    points = torch.cat((centers.unsqueeze(-2), corners), -2)
    points_h = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    for sample in range(batch):
        augmentation = lidar_augmentation[sample].to(
            device=points_h.device, dtype=points_h.dtype
        )
        raw = torch.einsum(
            "ij,kpj->kpi", torch.linalg.inv(augmentation), points_h[sample]
        ).detach()
        for view in range(views):
            projection = lidar2image[sample, view].to(
                device=raw.device, dtype=raw.dtype
            )
            camera = torch.einsum("ij,kpj->kpi", projection, raw)
            xy = camera[..., :2] / camera[..., 2:3].clamp_min(1e-5)
            # 原 decoder_utils.py:683 将 img_scale_factor 固定为 [1,1]；
            # 随后严格执行 crop，再依据 flip 前 img_shape 水平翻转。
            xy = xy - image_crop_offset[sample].to(
                device=xy.device, dtype=xy.dtype
            )
            if bool(image_flip[sample].item()):
                flip_width = image_shape[sample, 1].to(
                    device=xy.device, dtype=xy.dtype
                )
                xy = torch.stack((flip_width - xy[..., 0], xy[..., 1]), -1)
            current_image_size = image_size[sample].to(
                device=xy.device, dtype=xy.dtype
            )
            image_h, image_w = current_image_size
            on_image = (
                (xy[:, 0, 0] > 0) & (xy[:, 0, 0] < image_w)
                & (xy[:, 0, 1] > 0) & (xy[:, 0, 1] < image_h)
            )
            ids = torch.nonzero(on_image, as_tuple=False).flatten()
            if ids.numel() <= 1:
                continue
            projected_corners = xy[ids, 1:]
            rois = torch.stack(
                (torch.zeros_like(ids, dtype=image_feature.dtype),
                 projected_corners[..., 0].amin(-1),
                 projected_corners[..., 1].amin(-1),
                 projected_corners[..., 0].amax(-1),
                 projected_corners[..., 1].amax(-1)), -1
            )
            pooled[sample, view, ids] = roi_align(
                image_feature[sample, view].unsqueeze(0), rois,
                output_size=(output_size, output_size), spatial_scale=.25,
                sampling_ratio=sampling_ratio, aligned=True
            )
            visible[sample, view, ids] = True
    return pooled, visible, boxes


def _meta(image_feature, center, height, dimension, rotation, velocity,
          lidar2image, lidar_augmentation, image_size, image_shape,
          image_crop_offset, image_flip, point_cloud_range,
          voxel_size, out_size_factor: int, output_size: int, sampling_ratio: int):
    batch, views, channels = image_feature.shape[:3]
    proposals = center.shape[-1]
    return (
        image_feature.new_empty((batch, views, proposals, channels, output_size, output_size)),
        torch.empty((batch, views, proposals), device="meta", dtype=torch.bool),
        center.new_empty((batch, proposals, 9)),
    )


_IMPLEMENTATION = torch.library.Library(
    "fusionrepo", "IMPL", "CompositeExplicitAutograd"
)
_IMPLEMENTATION.impl("deepinteraction_image_roi_sampling", _impl)
_META = torch.library.Library("fusionrepo", "IMPL", "Meta")
_META.impl("deepinteraction_image_roi_sampling", _meta)


def deepinteraction_image_roi_sampling(*args):
    return torch.ops.fusionrepo.deepinteraction_image_roi_sampling(*args)
