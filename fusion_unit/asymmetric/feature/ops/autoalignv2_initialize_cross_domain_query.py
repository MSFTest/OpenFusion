import torch
from torch.library import Library

from utils import operator_registry as _registry
from fusion_unit.symmetric.feature.ops.feature_gate_multiply import feature_gate_multiply


def _impl(points, reference, assigned):
    """multiply_pts_detach 模式的跨域 query：参考图像乘以 detach 后的 voxel 特征。

    来源：multi_voxel_deformfusion.py::sample_single，400、409–410 行。
    Args:
        points: 浮点 [M,C] voxel 特征；reference: [1,Q,C] 参考图像特征；
            正式均 float32、C=64，同 device，正式 Q 对应 assigned 中的被选点数，
            eager 仍保留原乘法对单参考等输入的广播规则。
        assigned: 同 device bool [M]，按原体素行序取点。
    Returns:
        [1,Q,C] query，dtype 为两浮点分支的提升类型，device 不变。
        仅图像分支接收梯度，voxel 分支 detach；无 sigmoid、额外 FC 或内部状态。
        正式同形输入恢复 reference 的稠密排列，保留原 permute 后的非连续布局，
        避免公共门控的 contiguous 改变下游 CUDA Linear 计算路径。
        空输入保持原乘法广播语义，不合并重复坐标。reference 的 Q!=1 时可准确
        推导 Meta；Q=1 时原乘法可广播到 mask 决定的任意 Q，PyTorch 2.0 下明确
        拒绝 Meta/Fake，不伪造 Q=1，也不收紧原 eager 广播行为。
    """
    query = feature_gate_multiply(reference, points[assigned].unsqueeze(0).detach(), False)
    if query.shape == reference.shape:
        return torch.empty_like(reference, dtype=query.dtype).copy_(query)
    return query


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_initialize_cross_domain_query', _impl)


def _meta(points, reference, assigned):
    if reference.shape[1] == 1:
        raise RuntimeError('AutoAlignV2 单参考 query 广播依赖 mask 的动态 Q，不支持精确 Meta/Fake')
    shape = torch.broadcast_shapes(reference.shape, (1, reference.shape[1], points.shape[-1]))
    dtype = torch.promote_types(points.dtype, reference.dtype)
    if shape == reference.shape:
        return torch.empty_like(reference, dtype=dtype)
    return reference.new_empty(shape, dtype=dtype)


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_initialize_cross_domain_query', _meta)


def autoalignv2_initialize_cross_domain_query(points, reference, assigned):
    """调用 AutoAlignV2 跨域乘法 query；detach 和 Q 对齐约定见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_initialize_cross_domain_query(points, reference, assigned)
