"""UniTR 将图像 token 最近投影点量化为 LiDAR 网格坐标的算子。"""

import torch
from torch.library import Library

import utils.operator_registry  # noqa: F401


def _impl(nearest_xyz, pc_range, space_shape):
    sx, sy, _ = space_shape
    xyz = nearest_xyz.clone()
    xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0]) * sx - 0.5
    xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) * sy - 0.5
    xyz[..., 2] = 0.0
    xyz[..., 0] = torch.clamp(xyz[..., 0], min=0, max=sx - 1)
    xyz[..., 1] = torch.clamp(xyz[..., 1], min=0, max=sy - 1)
    return xyz[:, [2, 1, 0]].long()


def _meta(nearest_xyz, pc_range, space_shape):
    return nearest_xyz.new_empty((nearest_xyz.shape[0], 3), dtype=torch.int64)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("unitr_image_token_lidar_coordinates", _impl)
_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("unitr_image_token_lidar_coordinates", _meta)


def unitr_image_token_lidar_coordinates(nearest_xyz, pc_range, space_shape):
    """把最近投影点量化为 UniTR 图像到 LiDAR 的稀疏坐标。

    Args:
        nearest_xyz: [N,3] 浮点 LiDAR XYZ 坐标张量，单位为米。
        pc_range: 六个米制边界 [xmin,ymin,zmin,xmax,ymax,zmax]。
        space_shape: 三个网格边长 (sx,sy,sz)，用于 XY 轴量化。
    Returns:
        与输入同 device 的 int64 [N,3] 整数 ZYX 网格坐标；量化结果不可微。
        XY 先减半个 voxel 再 clamp 至 [0,sx-1]/[0,sy-1]，随后向零截断转整数；Z 固定为 0。
    """
    return torch.ops.fusionrepo.unitr_image_token_lidar_coordinates(
        nearest_xyz, list(pc_range), list(space_shape)
    )
