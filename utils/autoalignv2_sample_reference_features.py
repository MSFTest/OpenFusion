import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils.sample_features_by_grid_v2 import sample_features_by_grid_v2


def _impl(features, grid, assigned):
    """布尔选点后逐层采样并按原尺度顺序相加，生成参考图像特征。

    来源：multi_voxel_deformfusion.py::sample_single，372–386 行。
    Args:
        features: 非空 Tensor[]，每项同 dtype/device [1,C,H_l,W_l]，正式 C=64、L=3。
        grid: 同 dtype/device 浮点 [M,2]，[-1,1] 基准归一化坐标，可越界。
        assigned: 同 device bool [M]，保留被选点原行序，不排序/去重。
    Returns:
        同 dtype/device [1,Q,C]，Q=count_nonzero(assigned)。双线性、zero padding、
        align_corners=False；空 Q 返回 [1,0,C]，重复位置分别采样并累积图像梯度。
        对图像/grid 可微，mask 不可微；无参数、buffer 或训练状态。
        PyTorch 2.0 无准确数据依赖 Q 的 Meta/Fake 实现；注册明确拒绝的 Meta
        入口以阻止 FakeTensor 的真实零张量 fallback，绝不返回固定 Q 或上界。
    """
    if assigned.is_meta:
        raise NotImplementedError('AutoAlignV2 参考采样的动态 Q 在 PyTorch 2.0 不支持精确 Meta')
    valid_grid = grid[assigned].unsqueeze(0)
    result = 0.
    for feature in features:
        # v2 原语会转换 grid dtype，原项目 grid_sample 不允许该静默转换。
        if feature.dtype != grid.dtype:
            raise RuntimeError('AutoAlignV2 grid_sample 要求图像特征与 grid 的 dtype 一致')
        sampled = sample_features_by_grid_v2(feature, valid_grid, None, False, 'zeros').unsqueeze(-2)
        result += sampled
    return result.squeeze(-2).permute(0, 2, 1)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_sample_reference_features', _impl)


def _meta(features, grid, assigned):
    # 2.0 FakeTensor 会捕获 NotImplementedError 并用真实零 mask 跑 fallback，
    # 伪造 Q=0。RuntimeError 明确拒绝该路径，不是 shape 推导实现。
    raise RuntimeError('AutoAlignV2 参考采样的动态 Q 在 PyTorch 2.0 不支持精确 Meta/Fake')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_sample_reference_features', _meta)


def autoalignv2_sample_reference_features(features, grid, assigned):
    """调用 AutoAlignV2 参考采样；动态 Q 及 Fake 限制见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_sample_reference_features(features, grid, assigned)
