"""稀疏卷积私有 PyTorch 参考。依据 spconv.pytorch 的 SparseConvTensor、
SubMConv3d/SparseConv3d 调用契约及原项目 mmdet3d/ops/sparse_block.py
SparseBasicBlock 68–126、make_sparse_convmodule 129–191 行。原项目依赖
spconv 2.x；当前 CPU 环境无需 CUDA/C++ 即可执行等价的乘加与反传。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

try:
    from mmcv.ops import (
        SparseConvTensor as MMCVSparseConvTensor, SparseConv3d, SubMConv3d)
except ImportError:  # 正式构造时由下方给出明确错误
    MMCVSparseConvTensor = SparseConv3d = SubMConv3d = None


@dataclass
class SparseTensor:
    """[M,C] 特征与 int32 [M,4] (batch,z,y,x)，shape 为 (D,H,W)。"""
    features: torch.Tensor
    indices: torch.Tensor
    spatial_shape: tuple[int, int, int]
    batch_size: int

    def replace_feature(self, features: torch.Tensor) -> 'SparseTensor':
        return SparseTensor(features, self.indices, self.spatial_shape, self.batch_size)

    def to_native(self):
        if MMCVSparseConvTensor is None:
            raise RuntimeError("当前环境缺少 mmcv.ops sparse backend")
        return MMCVSparseConvTensor(
            self.features, self.indices.int(), self.spatial_shape, self.batch_size)

    @classmethod
    def from_native(cls, value):
        return cls(value.features, value.indices.int(), tuple(value.spatial_shape),
                   int(value.batch_size))

    def dense(self) -> torch.Tensor:
        """原 spconv.SparseConvTensor.dense 的 scatter_nd 写入语义。"""
        depth, height, width = self.spatial_shape
        output = self.features.new_zeros((self.batch_size * depth * height * width, self.features.shape[1]))
        if self.indices.numel():
            b, z, y, x = self.indices.long().unbind(1)
            linear = ((b * depth + z) * height + y) * width + x
            output[linear] = self.features
        return output.view(self.batch_size, depth, height, width, -1).permute(0, 4, 1, 2, 3).contiguous()


class SparseConv3dReference(nn.Module):
    """spconv2 权重布局 [out,kz,ky,kx,in]；索引坐标为 (b,z,y,x)。

    SubM 仅在输入已激活坐标输出；普通 sparse conv 在输入贡献到的所有
    合法输出坐标输出。每条 indice pair 执行 feat @ weight.T 并累加，
    PyTorch autograd 提供与原乘加公式一致的输入/权重梯度。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3,
                 stride=1, padding=0, subm=True):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.subm = subm
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        fan_in = in_channels * math.prod(self.kernel_size)
        nn.init.uniform_(self.weight, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))

    def forward(self, sparse: SparseTensor) -> SparseTensor:
        in_coords = [tuple(int(v) for v in row) for row in sparse.indices.tolist()]
        # spconv 的 CPU subm kernel 使用 unordered_map.insert：重复坐标
        # 只将首行放入邻居查找表；中心核位置始终是每行到自身。见本机
        # spconv/csrc/sparse/indices.py SpconvIndicesCPU.generate_subm_conv_inds。
        first_input = {}
        for i, key in enumerate(in_coords):
            first_input.setdefault(key, i)
        out_shape = tuple((s + 2 * p - k) // st + 1 for s, p, k, st in zip(
            sparse.spatial_shape, self.padding, self.kernel_size, self.stride))
        if self.subm:
            out_coords = in_coords
            out_shape = sparse.spatial_shape
        else:
            # 原 spconv/csrc/sparse/indices.py generate_conv_inds 先遍历
            # filter_offset，再按输入行序遍历；首次命中的输出坐标取得行号。
            active = {}
            for kz in range(self.kernel_size[0]):
                for ky in range(self.kernel_size[1]):
                    for kx in range(self.kernel_size[2]):
                        for b, z, y, x in in_coords:
                            numer = (z + self.padding[0] - kz,
                                     y + self.padding[1] - ky,
                                     x + self.padding[2] - kx)
                            if any(numer[a] % self.stride[a] for a in range(3)):
                                continue
                            coord = tuple(numer[a] // self.stride[a] for a in range(3))
                            if all(0 <= coord[a] < out_shape[a] for a in range(3)):
                                active.setdefault((b, *coord), None)
            out_coords = list(active)

        output = sparse.features.new_zeros((len(out_coords), self.weight.shape[0]))
        if self.subm:
            kernel_offsets = [(kz, ky, kx) for kz in range(self.kernel_size[0])
                              for ky in range(self.kernel_size[1])
                              for kx in range(self.kernel_size[2])]
            middle = len(kernel_offsets) // 2
            for flat, (kz, ky, kx) in enumerate(kernel_offsets[:middle + 1]):
                if flat == middle:
                    output = output + sparse.features @ self.weight[:, kz, ky, kx, :].T
                    continue
                sources, destinations = [], []
                for source, (b, z, y, x) in enumerate(in_coords):
                    target = (b, z - kz + self.padding[0],
                              y - ky + self.padding[1], x - kx + self.padding[2])
                    if target in first_input:
                        sources.append(source)
                        destinations.append(first_input[target])
                if sources:
                    src = torch.tensor(sources, device=output.device)
                    dst = torch.tensor(destinations, device=output.device)
                    contribution = sparse.features[src] @ self.weight[:, kz, ky, kx, :].T
                    output = output.index_add(0, dst, contribution)
                    rkz, rky, rkx = kernel_offsets[-1 - flat]
                    reverse = sparse.features[dst] @ self.weight[:, rkz, rky, rkx, :].T
                    output = output.index_add(0, src, reverse)
        else:
            output_map = {key: i for i, key in enumerate(out_coords)}
            for kz in range(self.kernel_size[0]):
                for ky in range(self.kernel_size[1]):
                    for kx in range(self.kernel_size[2]):
                        sources, destinations = [], []
                        for source, (b, z, y, x) in enumerate(in_coords):
                            numer = (z + self.padding[0] - kz,
                                     y + self.padding[1] - ky,
                                     x + self.padding[2] - kx)
                            if any(numer[a] % self.stride[a] for a in range(3)):
                                continue
                            target = (b, *[numer[a] // self.stride[a] for a in range(3)])
                            if target in output_map:
                                sources.append(source)
                                destinations.append(output_map[target])
                        if sources:
                            weighted = sparse.features[sources] @ self.weight[:, kz, ky, kx, :].T
                            output = output.index_add(0, torch.tensor(destinations, device=output.device), weighted)
        indices = torch.tensor(out_coords, dtype=torch.int32, device=sparse.indices.device).reshape(-1, 4)
        return SparseTensor(output, indices, out_shape, sparse.batch_size)


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


class SparseConvNormAct(nn.Module):
    """原 make_sparse_convmodule；正式路径使用当前 mmcv 原生 sparse op。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, subm=True, *, smoke_reference=False):
        super().__init__()
        self.smoke_reference = bool(smoke_reference)
        if self.smoke_reference:
            conv = SparseConv3dReference(
                in_channels, out_channels, kernel_size, stride, padding, subm)
        else:
            if SubMConv3d is None:
                raise RuntimeError("MSMDFusion 正式稀疏卷积需要 mmcv.ops")
            conv_type = SubMConv3d if subm else SparseConv3d
            conv = conv_type(in_channels, out_channels, kernel_size, stride=stride,
                             padding=padding, bias=False)
        self.add_module('0', conv)
        self.add_module('1', nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01))
        self.add_module('2', nn.ReLU(inplace=True))

    def forward(self, sparse: SparseTensor) -> SparseTensor:
        if self.smoke_reference:
            out = self._modules['0'](sparse)
        else:
            out = SparseTensor.from_native(self._modules['0'](sparse.to_native()))
        out = out.replace_feature(self._modules['1'](out.features))
        out = out.replace_feature(self._modules['2'](out.features))
        return out


class SparseBasicBlockReference(nn.Module):
    """原 SparseBasicBlock；由 ``smoke_reference`` 选择参考或原生卷积。"""

    def __init__(self, channels, *, smoke_reference=True):
        super().__init__()
        self.smoke_reference = bool(smoke_reference)
        if self.smoke_reference:
            self.conv1 = SparseConv3dReference(channels, channels, 3, padding=1)
            self.conv2 = SparseConv3dReference(channels, channels, 3, padding=1)
        else:
            if SubMConv3d is None:
                raise RuntimeError("MSMDFusion 正式稀疏卷积需要 mmcv.ops")
            self.conv1 = SubMConv3d(channels, channels, 3, padding=1, bias=False)
            self.conv2 = SubMConv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, sparse: SparseTensor) -> SparseTensor:
        identity = sparse.features
        if self.smoke_reference:
            out = self.conv1(sparse)
        else:
            out = SparseTensor.from_native(self.conv1(sparse.to_native()))
        out = out.replace_feature(self.relu(self.norm1(out.features)))
        if self.smoke_reference:
            out = self.conv2(out)
        else:
            out = SparseTensor.from_native(self.conv2(out.to_native()))
        out = out.replace_feature(self.norm2(out.features))
        return out.replace_feature(self.relu(out.features + identity))


def sparse_add(left: SparseTensor, right: SparseTensor) -> SparseTensor:
    """spconv.pytorch.functional.sparse_add：torch sparse COO 相加并 coalesce。"""
    shape = (left.batch_size, *left.spatial_shape, left.features.shape[1])
    a = torch.sparse_coo_tensor(left.indices.T.long(), left.features, shape, requires_grad=True)
    b = torch.sparse_coo_tensor(right.indices.T.long(), right.features, shape, requires_grad=True)
    summed = (a + b).coalesce()
    return SparseTensor(summed.values(), summed.indices().T.contiguous().int(), left.spatial_shape, left.batch_size)
