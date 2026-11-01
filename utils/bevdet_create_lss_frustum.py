# -*- coding: utf-8 -*-
"""LSS 图像视锥模板构造算子。"""

from __future__ import annotations

import torch
from torch.library import Library


def _create_lss_frustum_impl(
    depth_cfg: torch.Tensor, input_size: torch.Tensor, downsample: int
) -> torch.Tensor:
    """
    根据深度离散配置和图像尺寸生成 ``(u, v, depth)`` 视锥模板。

    Args:
        depth_cfg: 深度配置 ``(start, end, step)``，shape 为 ``(3,)``。
        input_size: 图像尺寸 ``(height, width)``，shape 为 ``(2,)``。
        downsample: 特征图相对输入图像的下采样倍数。

    Returns:
        Tensor: 图像像素坐标系视锥，shape 为 ``(D, Hf, Wf, 3)``。
    """
    if depth_cfg.numel() != 3 or input_size.numel() != 2:
        raise ValueError("depth_cfg must be (3,) and input_size must be (2,)")
    start, end, step = depth_cfg.to(torch.float32)
    height = int(input_size[0].item())
    width = int(input_size[1].item())
    feat_h = height // int(downsample)
    feat_w = width // int(downsample)
    depths = torch.arange(start, end, step, dtype=torch.float32, device=depth_cfg.device)
    depths = depths.view(-1, 1, 1).expand(-1, feat_h, feat_w)
    xs = torch.linspace(0, width - 1, feat_w, dtype=torch.float32, device=depth_cfg.device)
    ys = torch.linspace(0, height - 1, feat_h, dtype=torch.float32, device=depth_cfg.device)
    xs = xs.view(1, 1, feat_w).expand(depths.shape[0], feat_h, feat_w)
    ys = ys.view(1, feat_h, 1).expand(depths.shape[0], feat_h, feat_w)
    return torch.stack((xs, ys, depths), dim=-1).contiguous()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("create_lss_frustum", _create_lss_frustum_impl)


def _create_lss_frustum_meta(depth_cfg, input_size, downsample):
    start = float(depth_cfg[0].item())
    end = float(depth_cfg[1].item())
    step = float(depth_cfg[2].item())
    bins = max(int(torch.ceil(torch.tensor((end - start) / step)).item()), 0)
    height = int(input_size[0].item()) // int(downsample)
    width = int(input_size[1].item()) // int(downsample)
    return torch.empty((bins, height, width, 3), device="meta", dtype=torch.float32)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("create_lss_frustum", _create_lss_frustum_meta)


def create_lss_frustum(depth_cfg, input_size, downsample):
    """调用 ``fusionrepo::create_lss_frustum``。"""
    return torch.ops.fusionrepo.create_lss_frustum(depth_cfg, input_size, int(downsample))
