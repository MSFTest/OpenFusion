"""沿指定维取最大值及第一个最大索引，保留并列值反向传播语义。
Args: x 任意布局、浮点/整型 Tensor [...,R,...]；dim 归约轴（允许负轴）；keepdim 保留轴。
Returns: (values, indices)，归约后 shape 相同；values 跟随 x dtype/device，indices 为 int64。
空非归约维允许；空归约轴、非法轴沿用 torch.max 错误；不混合 device/dtype。
values 可微（并列只传给第一个最大值），indices 不可微；Meta/Fake 精确支持。
覆盖 TransFusion 高度压缩、3D-CVF LiDAR_Voxel_Feature_Encoding 的 voxel 内 max，
以及 F-ConvNet 已完成私有掩码处理后的 max 子功能；不承担 mask/all-invalid 策略。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry


def _impl(x, dim=1, keepdim=False):
    result = torch.max(x, dim=dim, keepdim=keepdim)
    return result.values, result.indices


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("indexed_max_reduce", _impl)


def _meta(x, dim=1, keepdim=False):
    result = torch.max(x, dim=dim, keepdim=keepdim)
    return result.values, result.indices


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("indexed_max_reduce", _meta)


def indexed_max_reduce(x, dim=1, keepdim=False):
    """沿指定维取最大值及第一个最大索引，保留并列值反向传播语义。
Args: x 任意布局、浮点/整型 Tensor [...,R,...]；dim 归约轴（允许负轴）；keepdim 保留轴。
Returns: (values, indices)，归约后 shape 相同；values 跟随 x dtype/device，indices 为 int64。
空非归约维允许；空归约轴、非法轴沿用 torch.max 错误；不混合 device/dtype。
values 可微（并列只传给第一个最大值），indices 不可微；Meta/Fake 精确支持。
覆盖 TransFusion 高度压缩、3D-CVF LiDAR_Voxel_Feature_Encoding 的 voxel 内 max，
以及 F-ConvNet 已完成私有掩码处理后的 max 子功能；不承担 mask/all-invalid 策略。
"""
    return torch.ops.fusionrepo.indexed_max_reduce(x, dim, keepdim)
