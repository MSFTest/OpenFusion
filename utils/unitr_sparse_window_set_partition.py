"""UniTR two-shift sparse window set partitioning."""

import math

import torch
from torch.library import Library


def _continuous_indices(setnum_per_win):
    set_num = int(setnum_per_win.sum().item())
    cumsum = torch.cumsum(setnum_per_win, 0)[:-1]
    set_win = torch.full((set_num,), 0, device=setnum_per_win.device, dtype=torch.long)
    set_win[cumsum] = 1
    set_win = torch.cumsum(set_win, 0)
    end_mask = (set_win - torch.roll(set_win, -1)) != 0
    template = torch.ones_like(set_win)
    template[end_mask] = (setnum_per_win - 1) * -1
    inner = torch.cumsum(template, 0)
    inner[end_mask] = setnum_per_win
    return set_win, inner - 1


def _window_indices(coords, sparse_shape, window_shape, shift):
    sx, sy, sz = sparse_shape
    wx, wy, wz = window_shape
    shift_x, shift_y, shift_z = shift
    if sz == wz:
        shift_z = 0
    cx = coords[:, 3] + shift_x
    cy = coords[:, 2] + shift_y
    cz = coords[:, 1] + shift_z
    nx, ny, nz = math.ceil(sx / wx) + 1, math.ceil(sy / wy) + 1, math.ceil(sz / wz) + 1
    win_x, win_y, win_z = cx // wx, cy // wy, cz // wz
    win = coords[:, 0] * (nx * ny * nz) + win_x * ny * nz + win_y * nz + win_z
    inner = torch.stack((cz % wz, cy % wy, cx % wx), -1)
    return win, inner


def _one_shift(coords, win_inds, inner, window_shape, set_size, expand_max_voxels):
    wx, wy, wz = window_shape
    max_voxel = wx * wy * wz * expand_max_voxels
    contiguous = torch.unique(win_inds, return_inverse=True)[1]
    per_win = torch.bincount(contiguous)
    win_num = per_win.shape[0]
    sets_per_win = torch.ceil(per_win / set_size).long()
    set_win, set_inner = _continuous_indices(sets_per_win)
    offsets = set_inner[:, None].repeat(1, set_size) * set_size
    base = torch.arange(set_size, device=coords.device)
    select = (offsets + base) * per_win[set_win, None]
    select = torch.floor(select.double() / (sets_per_win[set_win, None] * set_size).double())
    select = select + set_win[:, None] * max_voxel

    # Stable sort gives each point its input-order rank within its occupied window.
    inner_order = torch.empty_like(contiguous)
    if contiguous.numel():
        group_order = torch.argsort(contiguous, stable=True)
        group_starts = torch.cumsum(per_win, 0) - per_win
        sorted_ranks = (
            torch.arange(contiguous.numel(), device=coords.device)
            - torch.repeat_interleave(group_starts, per_win, output_size=contiguous.numel())
        )
        inner_order[group_order] = sorted_ranks
    global_order = contiguous * max_voxel + inner_order
    _, order1 = torch.sort(global_order)

    global_y = contiguous * max_voxel + inner[:, 1] * wx * wz + inner[:, 2] * wz + inner[:, 0]
    _, order2 = torch.sort(global_y)
    y_positions = -torch.ones_like(inner_order)
    y_positions.scatter_(0, order2, inner_order[order1])
    y_flat = torch.full((win_num * max_voxel,), -1, dtype=torch.long, device=coords.device)
    y_flat[y_positions + max_voxel * contiguous] = torch.arange(y_positions.numel(), device=coords.device)
    y_sets = y_flat[select.long()]

    global_x = contiguous * max_voxel + inner[:, 2] * wy * wz + inner[:, 1] * wz + inner[:, 0]
    _, order2 = torch.sort(global_x)
    x_positions = -torch.ones_like(inner_order)
    x_positions.scatter_(0, order2, inner_order[order1])
    x_flat = torch.full((win_num * max_voxel,), -1, dtype=torch.long, device=coords.device)
    x_flat[x_positions + max_voxel * contiguous] = torch.arange(x_positions.numel(), device=coords.device)
    x_sets = x_flat[select.long()]
    sets = torch.stack((y_sets, x_sets), 0)
    prefix = torch.roll(sets.clone(), shifts=1, dims=-1)
    prefix[..., 0] = -1
    mask = sets == prefix
    return sets, mask


def _impl(coords_bzyx, sparse_shape, window_shape, shift0, shift1, set_size: int, expand_max_voxels: int = 1):
    coords = coords_bzyx.long()
    outputs, masks, window_coords = [], [], []
    for shift in (shift0, shift1):
        win, inner = _window_indices(coords, sparse_shape, window_shape, shift)
        inds, mask = _one_shift(coords, win, inner, window_shape, set_size, expand_max_voxels)
        outputs.append(inds)
        masks.append(mask)
        window_coords.append(inner)
    return outputs[0], masks[0], window_coords[0], outputs[1], masks[1], window_coords[1]


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("unitr_sparse_window_set_partition", _impl)


def unitr_sparse_window_set_partition(
    coords_bzyx, sparse_shape, window_shape, shift0, shift1, set_size, expand_max_voxels=1
):
    """按 UniTR 的两个 shift 和 y/x 两种顺序，将 sparse token 分为等长局部集合。

    Args:
        coords_bzyx: (N,4)，整数或可转 long 的坐标，列为 (batch,z,y,x)；N 可为 0，可 non-contiguous。
        sparse_shape: 三个整数 (x,y,z)，源网格尺寸。
        window_shape: 三个正整数 (x,y,z)，窗口尺寸。
        shift0, shift1: 各为 (x,y,z) 整数窗口偏移。
        set_size: 每个 attention set 的 token 槽数。
        expand_max_voxels: 每窗索引槽位容量倍率，默认 1；容量为窗口体积乘倍率。

    Returns:
        六个张量，依次为 shift0 的 indices/mask/coors_in_win 和 shift1 的同类结果。indices 是
        (2,num_sets,set_size) long，沿输入 device，最前维 0/1 分别对应 y/x 轴排序，padding 值为 -1；
        mask 是同 shape bool，True 标记源算法定义的重复/填充位置。coors_in_win 是 (N,3) long，
        按输入行序输出窗口内 (z,y,x) 整数坐标。num_sets 随实际窗口占用变化。

    坐标与索引不可微；不会将坐标 clamp 到 sparse_shape。重复坐标仍是不同 token。动态 num_sets
    无准确 Meta kernel；槽位容量不足时保留源实现的越界错误，不动态扩容。来源：UniTR `dsvt.py:471-615`、
    `mm_backbone/unitr.py:615-630` 和 `model_utils/dsvt_utils.py:27-84,155-174`。
    """
    return torch.ops.fusionrepo.unitr_sparse_window_set_partition(
        coords_bzyx, list(sparse_shape), list(window_shape), list(shift0), list(shift1), set_size,
        expand_max_voxels,
    )
