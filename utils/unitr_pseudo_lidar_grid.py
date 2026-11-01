"""UniTR Image2Lidar 使用的静态 pseudo-LiDAR 采样网格算子。"""

import torch
from torch.library import Library

import utils.operator_registry  # noqa: F401


def _impl(pc_range, space_shape, sample_num, device="cpu"):
    sx, sy, sz = space_shape
    x1, y1, z1, x2, y2, z2 = pc_range
    coord_x = torch.linspace(0, sx - 1, sx).view(1, -1, 1, 1).repeat(1, 1, sy, sz)
    coord_y = torch.linspace(0, sy - 1, sy).view(1, 1, -1, 1).repeat(1, sx, 1, sz)
    coord_z = torch.linspace(0, sz - 1, sz).view(1, 1, 1, -1).repeat(1, sx, sy, 1)
    coords = torch.stack((coord_x, coord_y, coord_z), -1).view(-1, 3)
    points = coords.clone().float()
    points[..., 0] = ((points[..., 0] + 0.5) / sx) * (x2 - x1) + x1
    points[..., 1] = ((points[..., 1] + 0.5) / sy) * (y2 - y1) + y1
    points[..., 2] = ((points[..., 2] + 0.5) / sz) * (z2 - z1) + z1
    if sample_num == 1:
        return points.unsqueeze(1).to(device)
    points = points.unsqueeze(1).repeat(1, sample_num, 1)
    points[..., 2] = torch.linspace(z1, z2, sample_num).unsqueeze(0)
    return points.to(device)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("unitr_pseudo_lidar_grid", _impl)


def unitr_pseudo_lidar_grid(pc_range, space_shape, sample_num, device="cpu"):
    """构造完整 BEV pseudo-LiDAR 米制采样网格。

    Args:
        pc_range: 六个 LiDAR 米制边界 [xmin,ymin,zmin,xmax,ymax,zmax]。
        space_shape: 网格形状 (sx,sy,sz)，按 X、Y、Z 轴排列。
        sample_num: 每个网格位置的 Z 采样点数。
        device: 输出设备字符串，默认 "cpu"；计算按 UniTR 源码先在 CPU 构造再迁移。
    Returns:
        指定 device 上的 float32 [sx*sy*sz,sample_num,3] 米制 XYZ 网格，扁平顺序遵从连续 [sx,sy,sz] 布局；坐标不可微。
        XY 固定为 voxel 中心；sample_num==1 时 Z 为体素中心，否则 Z 等距覆盖包含端点的 [zmin,zmax]。
        本 schema 没有 Tensor 输入，PyTorch 2.0.1 无法选择 Meta dispatch key，因此不注册 Meta kernel；device="meta" 时由 Composite 实现实际构造 Meta 张量。
    """
    return torch.ops.fusionrepo.unitr_pseudo_lidar_grid(
        list(pc_range), list(space_shape), sample_num, str(device)
    )
