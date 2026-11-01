import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils._pointaugmenting_dispatch import register_no_grad
from utils.pixel_to_grid_normalized import pixel_to_grid_normalized


def _check(correspondence):
    if correspondence.dtype != torch.float32 or correspondence.ndim != 2 or correspondence.shape[1] != 3:
        raise ValueError('需要 float32 correspondence[N,3]')


def _impl(correspondence):
    """原地归一化像素及相机编号，返回共享存储的无梯度视图。

    来源：原 loading.py::LoadPointCloudFromFile.__call__ 224–229 行。
    Args:
        correspondence: float32 [N,3]，列为原图 u/v 像素和六相机 id，允许 N=0、
            非连续但可写且无内部重叠的布局；输入设备不限，计算显式在 CPU。
    Returns:
        同 shape/dtype/device 的 detached 别名，输入被原地修改，stride 保持原样。
        UV 复用 pixel_to_grid_normalized(height=896,width=1600)，相机列为 id/5*2-1。
        分母固定为 1599/895/5，不使用 448×800 或特征图尺寸，不 clamp -100 哨兵。
        无参数或状态，不可微；no_grad 本身不会清除输入叶子的 requires_grad。
    """
    _check(correspondence)
    uv = correspondence.detach().cpu().numpy()
    uv[:, :2] = pixel_to_grid_normalized(torch.from_numpy(uv[:, :2]), 896, 1600).numpy()
    uv[:, 2] = uv[:, 2] / 5 * 2 - 1
    correspondence.copy_(torch.from_numpy(uv).to(correspondence.device))
    return correspondence.detach()


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('pointaugmenting_normalize_sampling_coordinates', _impl)


def _meta(correspondence):
    """保持输入的 shape/stride/dtype 和别名关系，不读取坐标值。"""
    _check(correspondence)
    return correspondence.detach()


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('pointaugmenting_normalize_sampling_coordinates', _meta)
register_no_grad('pointaugmenting_normalize_sampling_coordinates')


def pointaugmenting_normalize_sampling_coordinates(correspondence):
    """调用原地三轴归一化算子；返回别名和完整输入输出契约见 _impl。"""
    return torch.ops.fusionrepo.pointaugmenting_normalize_sampling_coordinates(correspondence)
