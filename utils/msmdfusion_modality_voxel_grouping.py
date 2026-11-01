"""LiDAR/相机体素分组。原源码：mmdet3d/models/detectors/MSMDFusion.py，
type_assign 27–45 行、MSMDFusionDetector.voxel_modality_split 251–325 行。
"""

from __future__ import annotations

import torch
from torch.library import Library
from torch.nn import functional as F


def _impl(indices_3d: torch.Tensor, indices_2d: torch.Tensor,
          batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Args: 两组索引均为 int32 [M,4] (batch,z,y,x)，按原体素顺序。

    Returns: 两组 int32 [M,5] (batch,mix,z,y,x)，以及两组 long [K]
    的混合体素原序索引。使用完整整数坐标匹配；原源码十进制键在
    x/y>=1000 时碰撞，跨批偏移在 B>2 时重复，均由专项测试复现并修正。
    每批仍按排序及两指针匹配；索引不可微，动态 K 无准确 Meta。
    """
    all_3d, all_2d, syn_3d, syn_2d = [], [], [], []
    offset_3d, offset_2d = 0, 0
    for batch_id in range(batch_size):
        c3 = indices_3d[indices_3d[:, 0] == batch_id][:, 1:]
        c2 = indices_2d[indices_2d[:, 0] == batch_id][:, 1:]
        # 用完整整数坐标编码；原十进制常量在 y/x >= 1000 时发生碰撞。
        all_coords = torch.cat([c3, c2], dim=0)
        y_base = int(all_coords[:, 1].max().item()) + 1 if all_coords.numel() else 1
        x_base = int(all_coords[:, 2].max().item()) + 1 if all_coords.numel() else 1
        key3 = (c3[:, 0].long() * y_base + c3[:, 1].long()) * x_base + c3[:, 2].long()
        key2 = (c2[:, 0].long() * y_base + c2[:, 1].long()) * x_base + c2[:, 2].long()
        val3, perm3 = torch.sort(key3, dim=-1)
        val2, perm2 = torch.sort(key2, dim=-1)
        match3, match2 = torch.zeros_like(val3), torch.zeros_like(val2)
        i = j = 0
        # 原 type_assign 的顺序匹配；相等时双方各消耗一个。
        while i < val3.numel() and j < val2.numel():
            a, b = val3[i].item(), val2[j].item()
            if a < b:
                i += 1
            elif a == b:
                match3[i] = 1
                match2[j] = 1
                i += 1
                j += 1
            else:
                j += 1
        syn_3d.append(perm3[torch.where(match3)[0]] + offset_3d)
        syn_2d.append(perm2[torch.where(match2)[0]] + offset_2d)
        unsorted3 = torch.zeros_like(val3).scatter(-1, perm3, match3)
        unsorted2 = torch.zeros_like(val2).scatter(-1, perm2, match2)
        all_3d.append(F.pad(torch.cat([unsorted3[:, None], c3], dim=1), (1, 0), value=batch_id))
        all_2d.append(F.pad(torch.cat([unsorted2[:, None], c2], dim=1), (1, 0), value=batch_id))
        offset_3d += c3.shape[0]
        offset_2d += c2.shape[0]
    return (torch.cat(all_3d).int(), torch.cat(all_2d).int(),
            torch.cat(syn_3d), torch.cat(syn_2d))



_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("msmdfusion_modality_voxel_grouping", _impl)

def msmdfusion_modality_voxel_grouping(indices_3d: torch.Tensor, indices_2d: torch.Tensor,
                           batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按体素坐标同步 LiDAR 与相机行，标记混合体素。

    Args: 两组 int32 索引分别为 [M3,4]、[M2,4]，布局均为
        (batch,z,y,x)，单位为体素网格；batch_size 为批数。
    Returns: 两组 int32 [M,5] (batch,mix,z,y,x)，以及两组 long [K]
        的原行索引；保持输入 device。重复坐标按排序双指针一对一匹配，
        空批可返回空结果。K 数据依赖，PyTorch 2.0 无准确 Meta。
    """
    return torch.ops.fusionrepo.msmdfusion_modality_voxel_grouping(indices_3d, indices_2d, batch_size)
