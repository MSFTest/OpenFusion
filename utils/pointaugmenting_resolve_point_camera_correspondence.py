import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils._pointaugmenting_dispatch import check_numpy_float, register_no_grad
from common.indexed_max_reduce import indexed_max_reduce


def _check(pixels, camera_points):
    check_numpy_float(pixels, camera_points)
    if pixels.ndim != 3 or pixels.shape[0] != 6 or pixels.shape[2] != 2 or camera_points.shape != (6, pixels.shape[1], 3):
        raise ValueError('需要逐点对应的 pixels[6,N,2]、camera_points[6,N,3]')


def _impl(pixels, camera_points):
    """判定可见性并选择最后有效相机，保留逐点顺序和 float32 舍入边界。

    来源：原 loading.py::LoadPointCloudFromFile.__call__ 181–206 行。
    Args:
        pixels: float16/32/64 [6,N,2] 相机像素 uv；允许非连续和 N=0。
        camera_points: float16/32/64 [6,N,3] 相机 xyz，单位米，取 z 为深度。
        允许混合 dtype/device；同原 NumPy 预处理，显式复制到 CPU 比较与选择。
    Returns:
        float32 [N,3] (u,v,camera_id)，设备跟随 pixels；不可见三列均为 -100。
        严格 z>0、1<u<1599、1<v<895；uv 为 NaN/inf 时不命中。
        六相机顺序 0..5，重叠取最后命中，不按深度选择；重复点不合并。
        写成 float32 后才能进入归一化；不修改输入，无梯度、参数或状态。
    """
    _check(pixels, camera_points)
    uv = pixels.detach().cpu()
    xyz = camera_points.detach().cpu()
    valid = ((xyz[..., 2] > 0) & (uv[..., 0] > 1) & (uv[..., 0] < 1599)
             & (uv[..., 1] > 1) & (uv[..., 1] < 895))
    hit, reverse_index = indexed_max_reduce(valid.flip(0).to(torch.int64), 0, False)
    camera_id = 5 - reverse_index
    selected = uv[camera_id, torch.arange(uv.shape[1])]
    output = torch.cat((selected, camera_id[:, None]), dim=1).to(torch.float32)
    output = torch.where(hit[:, None].bool(), output, torch.full_like(output, -100))
    return output.to(pixels.device)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('pointaugmenting_resolve_point_camera_correspondence', _impl)


def _meta(pixels, camera_points):
    """对应表不删除点，静态输出 [N,3]，不读取可见点数。"""
    _check(pixels, camera_points)
    return torch.empty((pixels.shape[1], 3), dtype=torch.float32, device='meta')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('pointaugmenting_resolve_point_camera_correspondence', _meta)
register_no_grad('pointaugmenting_resolve_point_camera_correspondence')


def pointaugmenting_resolve_point_camera_correspondence(pixels, camera_points):
    """调用最后有效相机对应算子；开区间、哨兵及舍入契约见 _impl。"""
    return torch.ops.fusionrepo.pointaugmenting_resolve_point_camera_correspondence(pixels, camera_points)
