"""MVP 固定多源 split voxel block，原 DynamicVoxelEncoder(virtual=True)。
来源 det3d/models/readers/dynamic_voxel_encoder.py；保留无梯度、分组、inclusive范围、unique排序和均值顺序。
Args: points 为同设备float32[Ni,16]列表，xyz米、raw/painted/virtual tag在倒数第二列、秒在末列。
Returns: float32[V,21]、int64[V,4] batch-ZYX、int32 XYZ网格[1440,1440,40]。
空输入沿用原scatter/unique；动态V不宣称静态Meta兼容。0可学习参数。
"""
# The following code are copied from pytorch_scatter https://github.com/rusty1s/pytorch_scatter
# Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
# MIT License 
from typing import Optional, Tuple
import torch 

@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(other.dim()-src.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

@torch.jit.script
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:

    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)
    count = broadcast(count, out, dim)
    if torch.is_floating_point(out):
        out.div_(count)
    else:
        assert 0 
        # out.floor_divide_(count)
    return out
from torch import nn
from torch.nn import functional as F
import numpy as np
def voxelization_virtual(points, pc_range, voxel_size):
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[keep, :]
    real_points_mask = points[:, -2] == 1
    painted_points_mask = points[:, -2] == 0
    virtual_points_mask = points[:, -2] == -1
    real_points = points[real_points_mask][:, [0, 1, 2, 3, -1]]
    painted_point = points[painted_points_mask]
    virtual_point = points[virtual_points_mask]
    padded_points = torch.zeros(len(points), 22, device=points.device, dtype=points.dtype)
    padded_points[:len(real_points), :5] = real_points
    padded_points[:len(real_points), -1] = 1
    padded_points[len(real_points):len(real_points) + len(painted_point), 5:19] = painted_point[:, :-2]
    padded_points[len(real_points):len(real_points) + len(painted_point), 19] = painted_point[:, -1]
    padded_points[len(real_points):len(real_points) + len(painted_point), 20] = 1
    padded_points[len(real_points):len(real_points) + len(painted_point), 21] = 0
    padded_points[len(real_points) + len(painted_point):, 5:19] = virtual_point[:, :-2]
    padded_points[len(real_points) + len(painted_point):, 19] = virtual_point[:, -1]
    padded_points[len(real_points) + len(painted_point):, 20] = 0
    padded_points[len(real_points) + len(painted_point):, 21] = 0
    points_xyz = torch.cat([real_points[:, :3], painted_point[:, :3], virtual_point[:, :3]], dim=0)
    coords = ((points_xyz[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) / voxel_size[[2, 1, 0]]).to(torch.int64)
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)
    voxels = scatter_mean(padded_points, inverse_indices, dim=0)
    indicator = voxels[:, -1]
    mix_mask = (indicator > 0) * (indicator < 1)
    voxels = voxels[:, :-1]
    voxels[mix_mask, :5] = voxels[mix_mask, :5] / indicator[mix_mask].unsqueeze(-1)
    voxels[mix_mask, 5:] = voxels[mix_mask, 5:] / (1 - indicator[mix_mask].unsqueeze(-1))
    return (voxels, unique_coords)

class MVPDynamicVoxelEncoder(nn.Module):

    def __init__(self, pc_range=(-54.,-54.,-5.,54.,54.,3.), voxel_size=(.075,.075,.2), virtual=True):
        super(MVPDynamicVoxelEncoder, self).__init__()
        self.pc_range = torch.tensor(pc_range)
        self.voxel_size = torch.tensor(voxel_size)
        self.shape = torch.round((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size)
        self.shape_np = self.shape.numpy().astype(np.int32)
        self.virtual = virtual

    @torch.no_grad()
    def forward(self, points):
        coors = []
        voxels = []
        for res in points:
            if self.virtual:
                voxel, coor = voxelization_virtual(res, self.pc_range.to(res.device), self.voxel_size.to(res.device))
            else:
                voxel, coor = (_ for _ in ()).throw(ValueError('MVP split voxelization requires virtual=True'))
            voxels.append(voxel)
            coors.append(coor)
        coors_batch = []
        for i in range(len(voxels)):
            coor_pad = F.pad(coors[i], (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        voxels_batch = torch.cat(voxels, dim=0)
        return (voxels_batch, coors_batch, self.shape_np)
