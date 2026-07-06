# -*- coding: utf-8 -*-
"""
channel_reduce_1x1_2d

说明：
    将“二维特征图 1×1 通道压缩/映射”注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.channel_reduce_1x1_2d(
            x,
            weight,
            bias=None
        )

兼容性：
    - 兼容 PyTorch 2.0.1
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    对二维特征图执行 1×1 卷积，
    在不改变空间尺寸 (H, W) 的情况下改变通道数。

数学形式：
    对每个空间位置 (h, w)，执行：
        y[:, :, h, w] = W @ x[:, :, h, w] + b

输入：
    x: (B, C_in, H, W)
        输入二维特征图

    weight: (C_out, C_in)
        1×1 卷积权重
        等价于 Conv2d(C_in, C_out, kernel_size=1) 的权重去掉 1×1 维度后
        即可学习参数矩阵

    bias: (C_out,) 或 None
        偏置项，可选

输出：
    y: (B, C_out, H, W)
        输出二维特征图

等价替代：
    - BEVFeatureReduction1x1
    - ImageFeatureReduction1x1
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import Library




# ============================================================
# 2) Python 实现
# ============================================================
def _channel_reduce_1x1_2d_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    具体实现：
        x      : (B, C_in, H, W)
        weight : (C_out, C_in)
        bias   : (C_out,) 或 None
    """
    if x.dim() != 4:
        raise ValueError("x must have shape (B, C_in, H, W)")

    if weight.dim() != 2:
        raise ValueError("weight must have shape (C_out, C_in)")

    B, C_in, H, W = x.shape
    C_out, C_in_w = weight.shape

    if C_in_w != C_in:
        raise ValueError("weight.shape[1] must equal input channel C_in")

    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != C_out:
            raise ValueError("bias must have shape (C_out,)")

        bias = bias.to(device=x.device, dtype=x.dtype)

    weight = weight.to(device=x.device, dtype=x.dtype)

    # Conv2d 需要 (C_out, C_in, 1, 1)
    conv_weight = weight.view(C_out, C_in, 1, 1)

    y = F.conv2d(
        x,
        conv_weight,
        bias=bias,
        stride=1,
        padding=0
    )

    return y


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("channel_reduce_1x1_2d", _channel_reduce_1x1_2d_impl)


# ============================================================
# 3) Meta 实现
# ============================================================
def _channel_reduce_1x1_2d_meta(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must have shape (B, C_in, H, W)")

    if weight.dim() != 2:
        raise ValueError("weight must have shape (C_out, C_in)")

    B, _, H, W = x.shape
    C_out = weight.shape[0]

    return torch.empty(
        (B, C_out, H, W),
        device="meta",
        dtype=x.dtype
    )


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("channel_reduce_1x1_2d", _channel_reduce_1x1_2d_meta)


# ============================================================
# 4) Python 包装函数
# ============================================================
def channel_reduce_1x1_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.channel_reduce_1x1_2d

    Args:
        x: (B, C_in, H, W)
        weight: (C_out, C_in)
        bias: (C_out,) 或 None

    Returns:
        y: (B, C_out, H, W)
    """
    return torch.ops.fusionrepo.channel_reduce_1x1_2d(
        x,
        weight,
        bias
    )