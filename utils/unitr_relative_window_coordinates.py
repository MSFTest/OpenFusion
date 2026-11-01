"""UniTR 窗口内稀疏坐标的相对位置编码算子。"""

import torch
from torch.library import Library

import utils.operator_registry  # noqa: F401


def _impl(coors_in_win, sparse_shape, window_shape, normalize_pos=False):
    if len(window_shape) == 2:
        win_x, win_y = window_shape
        win_z = 0
        ndim = 2
    elif window_shape[-1] == 1 and sparse_shape[-1] == 1:
        win_x, win_y = window_shape[:2]
        win_z = 0
        ndim = 2
    else:
        win_x, win_y, win_z = window_shape
        ndim = 3
    z = coors_in_win[:, 0].float() - win_z / 2
    y = coors_in_win[:, 1].float() - win_y / 2
    x = coors_in_win[:, 2].float() - win_x / 2
    if normalize_pos:
        x = x / win_x * 2 * 3.1415
        y = y / win_y * 2 * 3.1415
        z = z / win_z * 2 * 3.1415
    return torch.stack((x, y) if ndim == 2 else (x, y, z), dim=-1)


def _meta(coors_in_win, sparse_shape, window_shape, normalize_pos=False):
    ndim = 2 if len(window_shape) == 2 or (window_shape[-1] == 1 and sparse_shape[-1] == 1) else 3
    return coors_in_win.new_empty((coors_in_win.shape[0], ndim), dtype=torch.float32)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("unitr_relative_window_coordinates", _impl)
_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("unitr_relative_window_coordinates", _meta)


def unitr_relative_window_coordinates(coors_in_win, sparse_shape, window_shape, normalize_pos=False):
    """计算 UniTR 窗口内 token 的中心化相对位置。

    Args:
        coors_in_win: [N,3] 整数 ZYX 窗口坐标张量。
        sparse_shape: 整数 (sx,sy,sz) 稀疏网格形状。
        window_shape: 整数窗口边长，按源码顺序 (wx,wy[,wz])。
        normalize_pos: 是否按窗口边长归一化并乘以 2*3.1415，默认 False。
    Returns:
        与输入同 device 的 float32 [N,2] (X,Y) 或 [N,3] (X,Y,Z) 坐标；输入坐标不可微。
        当窗口及稀疏网格 Z 边长均为 1 时输出二维坐标，否则输出三维坐标；不裁剪边界外坐标。
    """
    return torch.ops.fusionrepo.unitr_relative_window_coordinates(
        coors_in_win, list(sparse_shape), list(window_shape), normalize_pos
    )
