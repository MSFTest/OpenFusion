import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils.pixel_to_grid_normalized import pixel_to_grid_normalized


def _impl(uvd, height, width, already):
    """归一化像素、判断原有效性并按相机顺序首次分配体素。

    来源：multi_voxel_deformfusion.py::point_sample 81–92 及 sample_single 370–371。
    Args:
        uvd: 浮点 [M,3]，列为像素 u/v/投影深度，与 already 同 device。
        height/width: padding 后图像 H/W；already: 可变 bool [M] 已归属状态。
    Returns:
        grid [M,2] 与 uvd 同 dtype/device；valid、assigned bool [M]；
        原地更新并返回的 already（与输入别名）。归一化分母为 H/W 而非 H-1/W-1。
        保留 depth>0、y 两侧闭边界及源码缺少 x 上界的实际条件；不排除 padding。
        重复坐标不合并，先到相机占有；空输入保持空。grid 可微，布尔归属不可微。
        无内部状态/参数；相机之间显式传递 already，不保存为 buffer。
    """
    if height > 0 and width > 0:
        # 既有归一化原语分母是尺寸减一；这里只适配分母，不更改图像真实尺寸。
        grid = pixel_to_grid_normalized(uvd[:, :2], height + 1, width + 1)
    else:
        # 无效尺寸不在原语支持域；保留原源码张量除法的 inf/NaN 边界。
        grid = torch.cat([uvd[:, 0:1] / width * 2 - 1, uvd[:, 1:2] / height * 2 - 1], 1)
    x, y = grid[:, 0], grid[:, 1]
    valid = (uvd[:, -1] > 0) & (y >= -1) & (y <= 1) & (x >= -1) & (y <= 1)
    assigned = (~already) & valid
    already |= assigned
    return grid, valid, assigned, already


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_normalize_and_assign_camera', _impl)


def _meta(uvd, height, width, already):
    return _impl(uvd, height, width, already)


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_normalize_and_assign_camera', _meta)


def autoalignv2_normalize_and_assign_camera(uvd, height, width, already):
    """调用 AutoAlignV2 相机归属；原地别名及边界见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_normalize_and_assign_camera(uvd, height, width, already)
