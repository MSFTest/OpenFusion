# -*- coding: utf-8 -*-
"""基于通用 scatter 的纯 PyTorch BEVPoolV2 算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _bev_pool_v2_pytorch_impl(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_feat_shape: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    将深度概率与图像特征相乘，并按 ranks 聚合到 ego-BEV 网格。

    Args:
        depth: 深度概率，shape 为 ``(B,N,D,H,W)``。
        feat: 图像特征，shape 为 ``(B,N,C,H,W)``。
        ranks_depth: depth flatten 索引，shape 为 ``(M,)``。
        ranks_feat: feature flatten 索引，shape 为 ``(M,)``。
        ranks_bev: BEV voxel 线性索引，shape 为 ``(M,)``。
        bev_feat_shape: 输出形状 ``(B,Z,Y,X,C)``。
        interval_starts: BEVPoolV2 区间起点，shape 为 ``(Q,)``。
        interval_lengths: BEVPoolV2 区间长度，shape 为 ``(Q,)``。

    Returns:
        Tensor: Z 维折叠到通道后的 BEV 特征，shape 为 ``(B,C*Z,Y,X)``。
    """
    del interval_starts, interval_lengths
    bsz = int(bev_feat_shape[0].item())
    z_size = int(bev_feat_shape[1].item())
    y_size = int(bev_feat_shape[2].item())
    x_size = int(bev_feat_shape[3].item())
    channels = int(bev_feat_shape[4].item())
    if ranks_bev.numel() == 0:
        return feat.new_zeros((bsz, channels * z_size, y_size, x_size))

    feat_flat = feat.permute(0, 1, 3, 4, 2).reshape(-1, channels)
    weighted = feat_flat[ranks_feat.long()] * depth.reshape(-1)[ranks_depth.long(), None]
    rank = ranks_bev.long()
    x = torch.remainder(rank, x_size)
    rank = torch.div(rank, x_size, rounding_mode="floor")
    y = torch.remainder(rank, y_size)
    rank = torch.div(rank, y_size, rounding_mode="floor")
    z = torch.remainder(rank, z_size)
    batch = torch.div(rank, z_size, rounding_mode="floor")

    bev_slices = []
    for z_index in range(z_size):
        selected = z == z_index
        coords = torch.stack(
            [batch[selected], torch.zeros_like(batch[selected]), y[selected], x[selected]],
            dim=-1,
        ).to(torch.int32)
        bev_slices.append(
            torch.ops.fusionrepo.voxel_features_to_bev_scatter(
                weighted[selected], coords, bsz, y_size, x_size, "sum"
            )
        )
    return torch.cat(bev_slices, dim=1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("bev_pool_v2_pytorch", _bev_pool_v2_pytorch_impl)


def _bev_pool_v2_pytorch_meta(
    depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape,
    interval_starts, interval_lengths
):
    return torch.empty(
        (
            int(bev_feat_shape[0].item()),
            int(bev_feat_shape[4].item()) * int(bev_feat_shape[1].item()),
            int(bev_feat_shape[2].item()),
            int(bev_feat_shape[3].item()),
        ),
        device="meta",
        dtype=feat.dtype,
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("bev_pool_v2_pytorch", _bev_pool_v2_pytorch_meta)


def bev_pool_v2_pytorch(
    depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape,
    interval_starts, interval_lengths
):
    """调用 ``fusionrepo::bev_pool_v2_pytorch``。"""
    return torch.ops.fusionrepo.bev_pool_v2_pytorch(
        depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape,
        interval_starts, interval_lengths
    )
