# -*- coding: utf-8 -*-
"""
generate_depth_bins

说明：
    该文件将“深度 bins 生成”注册为 PyTorch 自定义算子：
        torch.ops.fusionrepo.generate_depth_bins(
            depth_range, num_bins, mode, bin_size, stride
        )

兼容性：
    - 兼容 PyTorch 2.0.1（不使用 torch.library.custom_op）
    - 使用 torch.library.Library 的 define/impl 机制注册

功能：
    根据给定深度范围与离散策略，生成深度 bins 的定义结果。
    该算子只负责“生成 bins 本身”，不包含任何任务相关后处理，
    例如：
        - 不做 HMFI 中的 (H, W, num_bins) 图像级展开
        - 不做 F-ConvNet 中的点到 bin 的分配(assign)

支持的生成模式：
    1. UD（Uniform Discretization）
       在 [d_min, d_max] 上均匀生成 num_bins 个深度位置

    2. LID（Linear Increasing Discretization）
       按 HMFI 原实现中的 LID 规则生成 num_bins 个深度位置

    3. SLIDING（Sliding Depth Bins）
       按 F-ConvNet 中 sliding frustum 的方式，
       根据 bin_size 与 stride 生成一组区间形式的深度 bins，
       每个 bin 表示为 [z_start, z_end]

模式编码建议：
    - 0: UD
    - 1: LID
    - 2: SLIDING

Args:
    depth_range: (2,)
        深度范围张量 [d_min, d_max]。

    num_bins: int
        深度 bins 数量。
        对于 UD / LID 模式必填。
        对于 SLIDING 模式可传占位值（例如 0），实际 bin 数由
        depth_range、bin_size、stride 自动决定。

    mode: int
        深度 bins 生成模式编码：
            - 0: UD
            - 1: LID
            - 2: SLIDING

    bin_size: float
        仅在 SLIDING 模式下使用，表示单个深度 bin 的长度。

    stride: float
        仅在 SLIDING 模式下使用，表示相邻深度 bin 的滑动步长。

Returns:
    depth_bins: Tensor
        - 当 mode 为 UD / LID 时：
            返回形状 (num_bins,) 的 1D 深度 bins 张量
        - 当 mode 为 SLIDING 时：
            返回形状 (T, 2) 的区间张量，
            每行为 [z_start, z_end]
"""

from __future__ import annotations

import torch
from torch.library import Library


# ============================================================
# 2) Python 实现（CompositeExplicitAutograd：用 PyTorch 张量算子实现，自动支持 autograd）
# ============================================================
def _generate_depth_bins_impl(
    depth_range: torch.Tensor,
    num_bins: int,
    mode: int,
    bin_size: float = 0.0,
    stride: float = 0.0
) -> torch.Tensor:
    """
    具体实现：
        - depth_range: (2,)
        - mode=0: UD     -> (num_bins,)
        - mode=1: LID    -> (num_bins,)
        - mode=2: sliding-> (T,2)
    """
    if depth_range.dim() != 1 or depth_range.numel() != 2:
        raise ValueError("depth_range must have shape (2,)")

    device = depth_range.device
    dtype = depth_range.dtype

    d_min = depth_range[0].to(dtype=dtype, device=device)
    d_max = depth_range[1].to(dtype=dtype, device=device)

    if d_max < d_min:
        raise ValueError("depth_range must satisfy d_max >= d_min")

    # --------------------------------------------------------
    # mode = 0 : UD
    # --------------------------------------------------------
    if mode == 0:
        if num_bins <= 0:
            raise ValueError("num_bins must be positive for UD mode")

        depth_bins = torch.linspace(
            d_min, d_max, num_bins, device=device, dtype=dtype
        )
        return depth_bins

    # --------------------------------------------------------
    # mode = 1 : LID
    # --------------------------------------------------------
    if mode == 1:
        if num_bins <= 0:
            raise ValueError("num_bins must be positive for LID mode")

        idx = torch.arange(num_bins, device=device, dtype=dtype)
        depth_bins = d_min + (d_max - d_min) * (idx * (idx + 1.0)) / (
            float(num_bins) * float(num_bins + 1)
        )
        return depth_bins

    # --------------------------------------------------------
    # mode = 2 : SLIDING
    # --------------------------------------------------------
    if mode == 2:
        if bin_size <= 0:
            raise ValueError("bin_size must be positive for SLIDING mode")
        if stride <= 0:
            raise ValueError("stride must be positive for SLIDING mode")

        span = torch.clamp(d_max - d_min - bin_size, min=0.0)
        T = int(torch.floor(span / stride).item()) + 1
        T = max(T, 1)

        starts = d_min + torch.arange(T, device=device, dtype=dtype) * stride
        ends = starts + bin_size
        depth_bins = torch.stack([starts, ends], dim=-1)   # (T, 2)
        return depth_bins

    raise ValueError(f"Unsupported depth binning mode code: {mode}")


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("generate_depth_bins", _generate_depth_bins_impl)


# ============================================================
# 3) Meta 实现（用于 shape 推导/编译/某些加速场景）
# ============================================================
def _generate_depth_bins_meta(
    depth_range: torch.Tensor,
    num_bins: int,
    mode: int,
    bin_size: float = 0.0,
    stride: float = 0.0
) -> torch.Tensor:
    dtype = depth_range.dtype

    if mode in (0, 1):
        if num_bins <= 0:
            num_bins = 1
        return torch.empty((num_bins,), device="meta", dtype=dtype)

    if mode == 2:
        # Meta 阶段无法可靠根据 item() 推真实 T，这里给保守占位形状
        return torch.empty((1, 2), device="meta", dtype=dtype)

    return torch.empty((1,), device="meta", dtype=dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("generate_depth_bins", _generate_depth_bins_meta)


# ============================================================
# 4) 可选：提供同名 Python 包装函数
# ============================================================
def generate_depth_bins(
    depth_range: torch.Tensor,
    num_bins: int,
    mode: int,
    bin_size: float = 0.0,
    stride: float = 0.0
) -> torch.Tensor:
    """
    Python 包装：内部调用 torch.ops.fusionrepo.generate_depth_bins

    注意：
        如果你项目里统一都用 torch.ops.fusionrepo.xxx 调用，则这个函数也可以不用。
    """
    return torch.ops.fusionrepo.generate_depth_bins(
        depth_range,
        int(num_bins),
        int(mode),
        float(bin_size),
        float(stride)
    )