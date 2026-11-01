# -*- coding: utf-8 -*-
"""SparseFusion BEV/图像融合坐标与 padding mask 构造算子。"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.library import Library

from utils.bevdet_compute_query_position import compute_query_position


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def _impl(valid_shapes: torch.Tensor, bev_h: int, bev_w: int,
          image_heights: List[int], image_widths: List[int], image_stride: int = 4
          ) -> Tuple[torch.Tensor, ...]:
    if valid_shapes.ndim != 3 or valid_shapes.shape[-1] != 2:
        raise ValueError("valid_shapes 必须是 [B,V,2]，末维为 (width,height)")
    if len(image_heights) != len(image_widths) or not image_heights:
        raise ValueError("图像层高宽必须非空且层数一致")
    if min(bev_h, bev_w, image_stride, *image_heights, *image_widths) <= 0:
        raise ValueError("所有空间尺寸和 stride 必须为正")
    device = valid_shapes.device
    indices = torch.arange(bev_h * bev_w, device=device, dtype=torch.long).unsqueeze(0)
    bev_one = compute_query_position(indices, bev_h, bev_w)
    bev_pos = bev_one.expand(valid_shapes.shape[0], -1, -1).contiguous()

    image_positions, normalized_positions, spatial, starts, masks = [], [], [], [], []
    offset = 0
    for level, (height, width) in enumerate(zip(image_heights, image_widths)):
        level_indices = torch.arange(height * width, device=device, dtype=torch.long).unsqueeze(0)
        position = compute_query_position(level_indices, height, width)
        image_positions.append(position)
        normalized_positions.append(_inverse_sigmoid(position / position.new_tensor((width, height))))
        spatial.append((height, width))
        starts.append(offset)
        offset += height * width
        divisor = float(image_stride * (2 ** level))
        valid_w = torch.floor(valid_shapes[..., 0] / divisor).long()
        valid_h = torch.floor(valid_shapes[..., 1] / divisor).long()
        yy = torch.arange(height, device=device).view(1, 1, height, 1)
        xx = torch.arange(width, device=device).view(1, 1, 1, width)
        masks.append(((yy >= valid_h[..., None, None]) | (xx >= valid_w[..., None, None])).reshape(
            valid_shapes.shape[0], valid_shapes.shape[1], -1
        ))
    return (
        bev_pos,
        torch.cat(image_positions, 1),
        torch.cat(normalized_positions, 1),
        torch.tensor(spatial, device=device, dtype=torch.long),
        torch.tensor(starts, device=device, dtype=torch.long),
        torch.cat(masks, -1),
    )


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("sparsefusion_fusion_coordinate_grid", _impl)


def _meta(valid_shapes, bev_h: int, bev_w: int, image_heights: List[int],
          image_widths: List[int], image_stride: int = 4):
    if valid_shapes.ndim != 3 or valid_shapes.shape[-1] != 2:
        raise ValueError("valid_shapes 必须是 [B,V,2]，末维为 (width,height)")
    if len(image_heights) != len(image_widths) or not image_heights:
        raise ValueError("图像层高宽必须非空且层数一致")
    if min(bev_h, bev_w, image_stride, *image_heights, *image_widths) <= 0:
        raise ValueError("所有空间尺寸和 stride 必须为正")
    total = sum(h * w for h, w in zip(image_heights, image_widths))
    batch, views = valid_shapes.shape[:2]
    float_opts = {"device": "meta", "dtype": torch.float32}
    return (
        torch.empty((batch, bev_h * bev_w, 2), **float_opts),
        torch.empty((1, total, 2), **float_opts),
        torch.empty((1, total, 2), **float_opts),
        torch.empty((len(image_heights), 2), device="meta", dtype=torch.long),
        torch.empty((len(image_heights),), device="meta", dtype=torch.long),
        torch.empty((batch, views, total), device="meta", dtype=torch.bool),
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("sparsefusion_fusion_coordinate_grid", _meta)


def sparsefusion_fusion_coordinate_grid(valid_shapes: torch.Tensor, bev_h: int, bev_w: int,
                                        image_heights: List[int], image_widths: List[int],
                                        image_stride: int = 4) -> Tuple[torch.Tensor, ...]:
    """生成 SparseFusion 融合坐标。

    ``valid_shapes`` 为 ``[B,V,2]``、末维 ``(有效宽,有效高)``，单位为增强后
    输入图像像素。返回 BEV cell-center ``[B,Hb*Wb,2]``、逐层图像中心及其
    inverse-sigmoid 坐标 ``[1,sum(Hl*Wl),2]``、``spatial_shapes[L,2]``、
    ``level_start_index[L]``、padding mask ``bool[B,V,sum]``。浮点坐标均为
    同 device 的 float32，mask 中 True 表示 padding；计算无可学习参数。
    """
    return torch.ops.fusionrepo.sparsefusion_fusion_coordinate_grid(
        valid_shapes, int(bev_h), int(bev_w), image_heights, image_widths, int(image_stride)
    )
