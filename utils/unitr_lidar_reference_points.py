"""UniTR 从体素坐标恢复 LiDAR 米制参考点的算子。"""

import torch
from torch.library import Library

import utils.operator_registry  # noqa: F401


def _impl(coords_bzyx, pc_range, space_shape, sample_num=1):
    sx, sy, sz = space_shape
    x1, y1, z1, x2, y2, z2 = pc_range
    coords = coords_bzyx[:, [3, 2, 1]]
    points = coords.clone().float()
    points[..., 0] = ((points[..., 0] + 0.5) / sx) * (x2 - x1) + x1
    points[..., 1] = ((points[..., 1] + 0.5) / sy) * (y2 - y1) + y1
    points[..., 2] = ((points[..., 2] + 0.5) / sz) * (z2 - z1) + z1
    if sample_num == 1:
        return points.unsqueeze(1)
    points = points.unsqueeze(1).repeat(1, sample_num, 1)
    points[..., 2] = torch.linspace(z1, z2, sample_num, device=points.device)
    return points


def _meta(coords_bzyx, pc_range, space_shape, sample_num=1):
    return coords_bzyx.new_empty((coords_bzyx.shape[0], sample_num, 3), dtype=torch.float32)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("unitr_lidar_reference_points", _impl)
_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("unitr_lidar_reference_points", _meta)


def unitr_lidar_reference_points(coords_bzyx, pc_range, space_shape, sample_num=1):
    """把体素网格位置变换为 LiDAR 米制参考点。

    Args:
        coords_bzyx: [N,4] 整数 (batch,z,y,x) 坐标张量。
        pc_range: 六个米制边界 [xmin,ymin,zmin,xmax,ymax,zmax]。
        space_shape: 三个网格边长 (sx,sy,sz)。
        sample_num: 每个体素采样点数，默认 1。
    Returns:
        与输入同 device 的 float32 [N,sample_num,3] XYZ 米制坐标，最后一维布局为 XYZ；坐标生成不可微。
        XY 使用体素中心，不对输入坐标裁剪；sample_num>1 时 Z 改为包含两端点的 [zmin,zmax] 等距采样。
    """
    return torch.ops.fusionrepo.unitr_lidar_reference_points(
        coords_bzyx, list(pc_range), list(space_shape), sample_num
    )
