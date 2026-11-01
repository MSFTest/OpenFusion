# -*- coding: utf-8 -*-
"""BEVDet dense heatmap 局部最大值筛选算子。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library


def _local_max_heatmap_nms_impl(
    heatmap: torch.Tensor, kernel_size: int, dataset_id: int
) -> torch.Tensor:
    """
    保留 BEV heatmap 局部峰值，并应用 nuScenes 类别例外规则。

    Args:
        heatmap: 已 sigmoid 的 heatmap，shape 为 ``(B,C,H,W)``。
        kernel_size: max-pool 局部窗口大小。
        dataset_id: ``0`` 表示 nuScenes，其类别 8/9 不做局部抑制。

    Returns:
        Tensor: 筛选后的 heatmap，shape 与输入一致。
    """
    padding = int(kernel_size) // 2
    if padding == 0:
        local_max = heatmap
    else:
        local_max = torch.zeros_like(heatmap)
        inner = F.max_pool2d(heatmap, kernel_size=int(kernel_size), stride=1, padding=0)
        local_max[:, :, padding:-padding, padding:-padding] = inner
    if int(dataset_id) == 0 and heatmap.shape[1] > 9:
        local_max[:, 8] = heatmap[:, 8]
        local_max[:, 9] = heatmap[:, 9]
    return (heatmap * (heatmap == local_max)).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("local_max_heatmap_nms", _local_max_heatmap_nms_impl)


def _local_max_heatmap_nms_meta(heatmap, kernel_size, dataset_id):
    return torch.empty(tuple(heatmap.shape), device="meta", dtype=heatmap.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("local_max_heatmap_nms", _local_max_heatmap_nms_meta)


def local_max_heatmap_nms(heatmap, kernel_size, dataset_id):
    """调用 ``fusionrepo::local_max_heatmap_nms``。"""
    return torch.ops.fusionrepo.local_max_heatmap_nms(
        heatmap, int(kernel_size), int(dataset_id)
    )
