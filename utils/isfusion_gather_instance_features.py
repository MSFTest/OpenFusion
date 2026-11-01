# -*- coding: utf-8 -*-
"""
IS-Fusion BEV scene feature 实例 token 收集算子。

Args:
    scene_feats: BEV scene feature，shape 为 (B, C, H, W)。
    topk_indices: BEV flatten index，shape 为 (B, K)，索引顺序为 y * W + x。

Returns:
    Tensor: instance tokens，shape 为 (B, C, K)。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _isfusion_gather_instance_features_impl(scene_feats: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    if scene_feats.dim() != 4:
        raise ValueError("scene_feats must have shape (B,C,H,W)")
    batch_size, channels = scene_feats.shape[:2]
    flat = scene_feats.flatten(2)
    index = topk_indices[:, None, :].expand(batch_size, channels, topk_indices.shape[1]).to(torch.long)
    return flat.gather(dim=2, index=index).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("isfusion_gather_instance_features", _isfusion_gather_instance_features_impl)


def _isfusion_gather_instance_features_meta(scene_feats: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    return torch.empty((scene_feats.shape[0], scene_feats.shape[1], topk_indices.shape[1]), device="meta", dtype=scene_feats.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("isfusion_gather_instance_features", _isfusion_gather_instance_features_meta)


def isfusion_gather_instance_features(scene_feats: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    return torch.ops.fusionrepo.isfusion_gather_instance_features(scene_feats, topk_indices)
