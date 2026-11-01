import torch
from torch.library import Library

from utils import operator_registry as _registry


def _impl(coors, voxel_size, point_cloud_range):
    """恢复增强 LiDAR 坐标系中的体素几何中心，不计算体素内点的质心。

    来源：原 multi_voxel_deformfusion.py::obtain_mlvl_feats，314–319 行。
    Args:
        coors: 同设备 int32/int64 [M,4]，列为 batch/z/y/x，允许非连续。
        voxel_size: 三个 XYZ 方向米制间距。
        point_cloud_range: 六个 XYZ 米制范围边界，使用前三个下界。
    Returns:
        同 device 浮点 [M,3] XYZ 米制坐标；dtype 遵循整数加 0.5 的
        PyTorch 默认浮点提升。不排序、不去重、不过滤越界；空输入为 [0,3]。
        离散坐标不可微；无参数、buffer 或 train/eval 状态。
    """
    x = (coors[:, 3] + 0.5) * voxel_size[0] + point_cloud_range[0]
    y = (coors[:, 2] + 0.5) * voxel_size[1] + point_cloud_range[1]
    z = (coors[:, 1] + 0.5) * voxel_size[2] + point_cloud_range[2]
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], -1)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_recover_voxel_centers', _impl)


def _meta(coors, voxel_size, point_cloud_range):
    return _impl(coors, voxel_size, point_cloud_range)


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_recover_voxel_centers', _meta)


def autoalignv2_recover_voxel_centers(coors, voxel_size, point_cloud_range):
    """调用 AutoAlignV2 体素中心恢复；参数、布局及边界见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_recover_voxel_centers(coors, voxel_size, point_cloud_range)
