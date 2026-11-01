import torch
from torch.library import Library

from utils import operator_registry as _registry
from common.sparsefusion_layers import ms_deform_attn_core_pytorch


def _impl(value, spatial_shapes, level_start_index, sampling_locations,
          attention_weights, im2col_step=64, backend='auto'):
    """AutoAlignV2 多尺度可变形注意力后端，复用已有 SparseFusion 数学 reference。

    来源：multi_voxel_deformfusion.py::sample_single 451–461 及
    ops/functions/ms_deform_attn_func.py；只包装后端，不新增采样公式。
    Args:
        value: 浮点 [B,S,H,D]，按 level 展平拼接的图像 value。
        spatial_shapes: 同 device int64 [L,2]，每层 (height,width)；
        level_start_index: int64 [L]，连续层起点；S=sum(height*width)。
        sampling_locations: 同 dtype/device [B,Q,H,L,P,2]，后端 [0,1] location
            约定；不修改调用者原样传入的坐标，不将 [-1,1] 静默转换。
        attention_weights: 同 device [B,Q,H,L,P] 或 [B,Q,H,L*P]；正式与 value
            同 dtype，显式 pytorch reference 保留原加权乘法的 dtype 提升。
        im2col_step: 原 CUDA 分块参数，默认 64；backend: auto/pytorch/mmcv。
    Returns:
        同 device [B,Q,H*D]；pytorch 路径 dtype 为 value/weight 的提升类型，
        MMCV 路径遵循扩展的 value dtype。双线性零填充、align_corners=False，按层/点
        加权后拼接 head。value/location/weight 可微，shape/index 不可微；Q=0 合法。
        CUDA 空查询直接构造空视图并连接三个输入的零梯度，避免原扩展启动零 block；
        仍先导入扩展检查依赖，非空查询始终调用原 CUDA kernel。
        auto 保持原 CPU reference / CUDA MMCV 分支；显式 mmcv 缺扩展直接报错，
        不捕获依赖错误后换算法。auto 的混合 dtype Meta 无法区分 CPU 提升与
        CUDA 扩展转换，因此明确拒绝，调用者可显式指定 pytorch/mmcv。无参数或训练状态。
    """
    if backend not in ('auto', 'pytorch', 'mmcv'):
        raise ValueError('backend 必须是 auto、pytorch 或 mmcv')
    if backend == 'mmcv' or (backend == 'auto' and value.is_cuda and torch.cuda.is_available()):
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
        if value.is_cuda and sampling_locations.shape[1] == 0:
            empty = value[:, :0].flatten(2)
            return empty + sampling_locations.to(value.dtype).sum() * 0 + attention_weights.to(value.dtype).sum() * 0
        return MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step)
    return ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_deform_attention_backend', _impl)


def _meta(value, spatial_shapes, level_start_index, sampling_locations,
          attention_weights, im2col_step=64, backend='auto'):
    if backend not in ('auto', 'pytorch', 'mmcv'):
        raise ValueError('backend 必须是 auto、pytorch 或 mmcv')
    if backend == 'auto' and (value.dtype != attention_weights.dtype or value.dtype != sampling_locations.dtype):
        raise RuntimeError('auto 后端的混合 dtype Meta/Fake 需显式指定 pytorch 或 mmcv')
    if backend == 'pytorch' and value.dtype != sampling_locations.dtype:
        raise RuntimeError('pytorch grid_sample 要求 value 与 sampling_locations dtype 一致')
    dtype = value.dtype if backend == 'mmcv' else torch.promote_types(value.dtype, attention_weights.dtype)
    return value.new_empty((value.shape[0], sampling_locations.shape[1], value.shape[2] * value.shape[3]), dtype=dtype)


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_deform_attention_backend', _meta)


def autoalignv2_deform_attention_backend(value, spatial_shapes, level_start_index, sampling_locations,
                                        attention_weights, im2col_step=64, backend='auto'):
    """调用已注册 AutoAlignV2 后端；形状、坐标及依赖失败约定见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_deform_attention_backend(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step, backend)
