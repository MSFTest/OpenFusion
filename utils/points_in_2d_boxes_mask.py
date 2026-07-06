# -*- coding: utf-8 -*-
"""
points_in_2d_boxes_mask

说明：
    判断图像平面点是否位于 2D 检测框内，可选同时约束点位于图像边界内。
    该算子用于 frustum 构造、点语义查询、图像 ROI 预筛选等二维几何过滤场景。

Args:
    points_2d: (N, 2) 或 (B, N, 2)
        图像像素坐标，最后一维为 (u, v)。

    boxes_2d: (4)、(K, 4)、(B, 4) 或 (B, K, 4)
        2D 框，格式为 [x1, y1, x2, y2]。

    image_width: int
        图像宽度。

    image_height: int
        图像高度。

    use_image_bounds: bool
        是否同时要求点落在 [0, image_width) 和 [0, image_height) 内。

Returns:
    mask: Tensor
        当 boxes_2d 为单框时，返回 (N,) 或 (B, N)；
        当 boxes_2d 为多框时，返回 (K, N) 或 (B, K, N)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _points_in_2d_boxes_mask_impl(
    points_2d: torch.Tensor,
    boxes_2d: torch.Tensor,
    image_width: int,
    image_height: int,
    use_image_bounds: bool = True,
) -> torch.Tensor:
    if points_2d.shape[-1] != 2:
        raise ValueError("points_2d last dim must be 2")
    if boxes_2d.shape[-1] != 4:
        raise ValueError("boxes_2d last dim must be 4")

    single_points = points_2d.dim() == 2
    if single_points:
        pts = points_2d.unsqueeze(0)
    else:
        pts = points_2d

    if pts.dim() != 3:
        raise ValueError("points_2d must have shape (N, 2) or (B, N, 2)")

    B, N, _ = pts.shape
    u = pts[..., 0]
    v = pts[..., 1]

    boxes = boxes_2d.to(device=pts.device, dtype=pts.dtype)
    single_box = boxes.dim() == 1

    if single_box:
        boxes = boxes.view(1, 1, 4).expand(B, 1, 4)
    elif boxes.dim() == 2:
        if boxes.shape[0] == B and not single_points:
            boxes = boxes.view(B, 1, 4)
            single_box = True
        else:
            boxes = boxes.unsqueeze(0).expand(B, boxes.shape[0], 4)
    elif boxes.dim() == 3:
        if boxes.shape[0] != B:
            raise ValueError("batched boxes_2d must share batch size with points_2d")
    else:
        raise ValueError("boxes_2d must have shape (4), (K,4), (B,4), or (B,K,4)")

    x1 = boxes[..., 0].unsqueeze(-1)
    y1 = boxes[..., 1].unsqueeze(-1)
    x2 = boxes[..., 2].unsqueeze(-1)
    y2 = boxes[..., 3].unsqueeze(-1)

    uu = u.unsqueeze(1)
    vv = v.unsqueeze(1)
    mask = (uu >= x1) & (uu < x2) & (vv >= y1) & (vv < y2)

    if use_image_bounds:
        in_img = (uu >= 0) & (uu < int(image_width)) & (vv >= 0) & (vv < int(image_height))
        mask = mask & in_img

    if single_box:
        mask = mask.squeeze(1)
    if single_points:
        mask = mask.squeeze(0)
    return mask


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("points_in_2d_boxes_mask", _points_in_2d_boxes_mask_impl)


def _points_in_2d_boxes_mask_meta(
    points_2d: torch.Tensor,
    boxes_2d: torch.Tensor,
    image_width: int,
    image_height: int,
    use_image_bounds: bool = True,
) -> torch.Tensor:
    if points_2d.dim() == 2:
        n = points_2d.shape[0]
        if boxes_2d.dim() == 1:
            return torch.empty((n,), device="meta", dtype=torch.bool)
        return torch.empty((boxes_2d.shape[0], n), device="meta", dtype=torch.bool)

    b, n, _ = points_2d.shape
    if boxes_2d.dim() in (1, 2):
        return torch.empty((b, n), device="meta", dtype=torch.bool)
    return torch.empty((b, boxes_2d.shape[1], n), device="meta", dtype=torch.bool)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("points_in_2d_boxes_mask", _points_in_2d_boxes_mask_meta)


def points_in_2d_boxes_mask(
    points_2d: torch.Tensor,
    boxes_2d: torch.Tensor,
    image_width: int,
    image_height: int,
    use_image_bounds: bool = True,
) -> torch.Tensor:
    return torch.ops.fusionrepo.points_in_2d_boxes_mask(
        points_2d,
        boxes_2d,
        int(image_width),
        int(image_height),
        bool(use_image_bounds),
    )
