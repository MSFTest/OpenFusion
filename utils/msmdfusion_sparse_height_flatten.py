"""末级稀疏体素密集化并折叠高度。原源码：mmdet3d/models/detectors/
MSMDFusion.py，MSMDFusionDetector.extract_pts_feat 440–445 行；spconv2
SparseConvTensor.dense 与 scatter_nd 的索引写入语义。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _impl(features: torch.Tensor, indices: torch.Tensor, batch_size: int,
          depth: int, height: int, width: int) -> torch.Tensor:
    """Args: features [M,C]；indices int32 [M,4] (batch,z,y,x)。

    Returns: [B,C*D,H,W] BEV，C 后按 z 序展开。原 spconv dense 先将
    稀疏特征写入 [B,D,H,W,C]，重复索引直接覆盖，之后转为
    [B,C,D,H,W] contiguous 并 view。输入特征可微；索引不可微。
    越界索引由 PyTorch 索引规则处理，不预先裁剪。
    """
    channels = features.shape[1]
    dense = features.new_zeros((batch_size * depth * height * width, channels))
    b, z, y, x = indices.long().unbind(1)
    linear = ((b * depth + z) * height + y) * width + x
    dense[linear] = features
    dense = dense.view(batch_size, depth, height, width, channels)
    dense = dense.permute(0, 4, 1, 2, 3).contiguous()
    return dense.view(batch_size, channels * depth, height, width)


def _meta(features: torch.Tensor, indices: torch.Tensor, batch_size: int,
          depth: int, height: int, width: int) -> torch.Tensor:
    return torch.empty((batch_size, features.shape[1] * depth, height, width),
                       dtype=features.dtype, device='meta')



_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("msmdfusion_sparse_height_flatten", _impl)

_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("msmdfusion_sparse_height_flatten", _meta)

def msmdfusion_sparse_height_flatten(features: torch.Tensor, indices: torch.Tensor, batch_size: int,
                          depth: int, height: int, width: int) -> torch.Tensor:
    """将稀疏体素密集化后把高度 z 折入 BEV 通道。

    Args: features 为 [M,C]；indices 为 int32 [M,4] (batch,z,y,x)
        体素坐标；四个整数为 B,D,H,W。
    Returns: [B,C*D,H,W]，与 features 同 dtype/device。重复坐标
        后写覆盖，越界不静默裁剪；索引无梯度，特征路径可微。
    """
    return torch.ops.fusionrepo.msmdfusion_sparse_height_flatten(
        features, indices, batch_size, depth, height, width)
