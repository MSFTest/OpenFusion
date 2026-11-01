# -*- coding: utf-8 -*-
"""
IS-Fusion scene tokens 到 dense BEV feature map 的恢复算子。

Args:
    scene_tokens: scene tokens，shape 为 (B, C, H*W) 或 (B, H*W, C)。
    bev_h: BEV 高度。
    bev_w: BEV 宽度。

Returns:
    Tensor: BEV 特征图，shape 为 (B, C, bev_h, bev_w)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _isfusion_reshape_scene_tokens_to_bev_impl(scene_tokens: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    if scene_tokens.dim() != 3:
        raise ValueError("scene_tokens must have shape (B,C,H*W) or (B,H*W,C)")
    batch_size = scene_tokens.shape[0]
    token_count = int(bev_h) * int(bev_w)
    if scene_tokens.shape[2] == token_count:
        channels = scene_tokens.shape[1]
        return scene_tokens.reshape(batch_size, channels, int(bev_h), int(bev_w)).contiguous()
    if scene_tokens.shape[1] == token_count:
        channels = scene_tokens.shape[2]
        return scene_tokens.transpose(1, 2).reshape(batch_size, channels, int(bev_h), int(bev_w)).contiguous()
    raise ValueError("one scene_tokens dimension must equal bev_h * bev_w")


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("isfusion_reshape_scene_tokens_to_bev", _isfusion_reshape_scene_tokens_to_bev_impl)


def _isfusion_reshape_scene_tokens_to_bev_meta(scene_tokens: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    token_count = int(bev_h) * int(bev_w)
    channels = scene_tokens.shape[1] if scene_tokens.shape[2] == token_count else scene_tokens.shape[2]
    return torch.empty((scene_tokens.shape[0], channels, int(bev_h), int(bev_w)), device="meta", dtype=scene_tokens.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("isfusion_reshape_scene_tokens_to_bev", _isfusion_reshape_scene_tokens_to_bev_meta)


def isfusion_reshape_scene_tokens_to_bev(scene_tokens: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    return torch.ops.fusionrepo.isfusion_reshape_scene_tokens_to_bev(scene_tokens, int(bev_h), int(bev_w))
