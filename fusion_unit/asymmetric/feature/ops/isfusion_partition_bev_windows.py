# -*- coding: utf-8 -*-
"""
IS-Fusion G2R BEV token 窗口划分算子。

Args:
    grid_features: BEV token 特征，shape 为 (N, C)。
    grid_coords: BEV token 坐标，shape 为 (N, 4)，顺序为 (batch, z, y, x)。
    window_size: 窗口边长。
    shift_size: shifted-window 偏移量，为 0 时表示普通窗口。

Returns:
    tuple[Tensor, Tensor, Tensor]:
        - window_tokens: 窗口 token，shape 为 (num_windows, window_size*window_size, C)。
        - window_mask: padding 掩码，shape 为 (num_windows, window_size*window_size)。
        - restore_indices: token 在窗口展开后的恢复索引，shape 为 (N,)。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.library import Library


def _isfusion_partition_bev_windows_impl(
    grid_features: torch.Tensor,
    grid_coords: torch.Tensor,
    window_size: int,
    shift_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grid_features.dim() != 2 or grid_coords.dim() != 2 or grid_coords.shape[1] != 4:
        raise ValueError("grid_features must be (N,C) and grid_coords must be (N,4)")
    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")
    num_tokens, channels = grid_features.shape
    tokens_per_window = int(window_size) * int(window_size)
    if num_tokens == 0:
        empty_tokens = grid_features.new_zeros((0, tokens_per_window, channels))
        empty_mask = torch.ones((0, tokens_per_window), dtype=torch.bool, device=grid_features.device)
        empty_restore = torch.empty((0,), dtype=torch.long, device=grid_features.device)
        return empty_tokens, empty_mask, empty_restore

    coords = grid_coords.to(torch.long)
    shifted_y = coords[:, 2] + int(shift_size)
    shifted_x = coords[:, 3] + int(shift_size)
    batch = coords[:, 0]
    win_y = torch.div(shifted_y, int(window_size), rounding_mode="floor")
    win_x = torch.div(shifted_x, int(window_size), rounding_mode="floor")
    local_y = torch.remainder(shifted_y, int(window_size))
    local_x = torch.remainder(shifted_x, int(window_size))
    slot = local_y * int(window_size) + local_x

    max_win_y = win_y.max() + 1
    max_win_x = win_x.max() + 1
    packed_window = batch * (max_win_y * max_win_x) + win_y * max_win_x + win_x
    _, inverse = torch.unique(packed_window, sorted=True, return_inverse=True)
    num_windows = int(inverse.max().item()) + 1

    window_tokens = grid_features.new_zeros((num_windows, tokens_per_window, channels))
    window_mask = torch.ones((num_windows, tokens_per_window), dtype=torch.bool, device=grid_features.device)
    restore_indices = inverse * tokens_per_window + slot
    flat = window_tokens.reshape(num_windows * tokens_per_window, channels)
    flat.index_copy_(0, restore_indices, grid_features)
    window_mask.reshape(-1).index_fill_(0, restore_indices, False)
    return window_tokens.contiguous(), window_mask.contiguous(), restore_indices.to(torch.long).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("isfusion_partition_bev_windows", _isfusion_partition_bev_windows_impl)


def _isfusion_partition_bev_windows_meta(
    grid_features: torch.Tensor,
    grid_coords: torch.Tensor,
    window_size: int,
    shift_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, channels = grid_features.shape
    tokens_per_window = int(window_size) * int(window_size)
    num_windows = (num_tokens + tokens_per_window - 1) // tokens_per_window
    return (
        torch.empty((num_windows, tokens_per_window, channels), device="meta", dtype=grid_features.dtype),
        torch.empty((num_windows, tokens_per_window), device="meta", dtype=torch.bool),
        torch.empty((num_tokens,), device="meta", dtype=torch.long),
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("isfusion_partition_bev_windows", _isfusion_partition_bev_windows_meta)


def isfusion_partition_bev_windows(
    grid_features: torch.Tensor,
    grid_coords: torch.Tensor,
    window_size: int,
    shift_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.fusionrepo.isfusion_partition_bev_windows(
        grid_features,
        grid_coords,
        int(window_size),
        int(shift_size),
    )
