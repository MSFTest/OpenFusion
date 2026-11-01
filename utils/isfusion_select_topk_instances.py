# -*- coding: utf-8 -*-
"""
IS-Fusion instance heatmap Top-K 选择算子。

Args:
    ins_heatmap: 实例中心 heatmap logits，shape 为 (B, num_classes, H, W)。
    bev_pos: BEV 位置表，shape 为 (B, H*W, 2)。
    topk: 每帧保留实例数量。
    nms_kernel_size: 局部最大值抑制窗口大小。

Returns:
    tuple[Tensor, Tensor, Tensor]:
        - topk_indices: BEV flatten index，shape 为 (B, K)。
        - query_pos: Top-K 实例位置，shape 为 (B, K, 2)。
        - topk_scores: Top-K heatmap 分数，shape 为 (B, K)。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.library import Library


def _isfusion_select_topk_instances_impl(
    ins_heatmap: torch.Tensor,
    bev_pos: torch.Tensor,
    topk: int,
    nms_kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if ins_heatmap.dim() != 4:
        raise ValueError("ins_heatmap must have shape (B,num_classes,H,W)")
    batch_size, num_classes, height, width = ins_heatmap.shape
    heatmap = ins_heatmap.sigmoid()
    if nms_kernel_size > 1:
        padding = int(nms_kernel_size) // 2
        local_max = F.max_pool2d(heatmap, kernel_size=int(nms_kernel_size), stride=1, padding=padding)
        heatmap = heatmap * (heatmap == local_max)
    flat = heatmap.reshape(batch_size, num_classes, height * width)
    k = min(int(topk), num_classes * height * width)
    topk_scores, topk_all = torch.topk(flat.reshape(batch_size, -1), k=k, dim=-1)
    topk_indices = torch.remainder(topk_all, height * width).to(torch.long)
    if k < int(topk):
        pad_count = int(topk) - k
        topk_indices = torch.cat([topk_indices, topk_indices[:, -1:].expand(-1, pad_count)], dim=1)
        topk_scores = torch.cat([topk_scores, topk_scores[:, -1:].expand(-1, pad_count)], dim=1)
    query_pos = bev_pos.gather(1, topk_indices[:, :, None].expand(-1, -1, bev_pos.shape[-1]))
    query_pos = torch.stack([query_pos[..., 1], query_pos[..., 0]], dim=-1)
    return topk_indices.contiguous(), query_pos.contiguous(), topk_scores.contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("isfusion_select_topk_instances", _isfusion_select_topk_instances_impl)


def _isfusion_select_topk_instances_meta(
    ins_heatmap: torch.Tensor,
    bev_pos: torch.Tensor,
    topk: int,
    nms_kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = ins_heatmap.shape[0]
    return (
        torch.empty((batch_size, int(topk)), device="meta", dtype=torch.long),
        torch.empty((batch_size, int(topk), 2), device="meta", dtype=bev_pos.dtype),
        torch.empty((batch_size, int(topk)), device="meta", dtype=ins_heatmap.dtype),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("isfusion_select_topk_instances", _isfusion_select_topk_instances_meta)


def isfusion_select_topk_instances(
    ins_heatmap: torch.Tensor,
    bev_pos: torch.Tensor,
    topk: int,
    nms_kernel_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.fusionrepo.isfusion_select_topk_instances(
        ins_heatmap,
        bev_pos,
        int(topk),
        int(nms_kernel_size),
    )
