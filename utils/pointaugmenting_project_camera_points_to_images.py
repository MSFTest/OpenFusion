import numpy as np
import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils._pointaugmenting_dispatch import check_numpy_float, register_no_grad


def _check(camera_points, intrinsics):
    check_numpy_float(camera_points, intrinsics)
    if camera_points.ndim != 3 or camera_points.shape[0] != 6 or camera_points.shape[2] != 3 or intrinsics.shape != (6, 3, 3):
        raise ValueError('需要 camera_points[6,N,3]、intrinsics[6,3,3]')


def _impl(camera_points, intrinsics):
    """六相机米制点投影到各自像素平面，保留无 epsilon 的 NumPy 除法。

    来源：原 loading.py::view_points 100–136 行及 __call__ 的 normalize=True 调用。
    Args:
        camera_points: float16/32/64 [6,N,3] 相机 xyz，单位米；可空、非连续。
        intrinsics: float16/32/64 [6,3,3] 相机内参，左乘列向量得到像素齐次坐标。
        允许混合 dtype/device，显式在 CPU 上使用原 float64 eye/ones 计算。
    Returns:
        float64 [6,N,2] 像素 (u,v)，设备跟随 camera_points，不可微。
        保留 NumPy stack 布局：N>1 时 stride=(2*N,1,N)，N=0/1 使用原退化布局。
        不加 epsilon、不 clamp、不筛选；零深度保留 inf/NaN，负深度照常相除。
        不修改输入；无参数、buffer 或 train/eval 状态，不包含可见性或融合。
    """
    _check(camera_points, intrinsics)
    xyz = camera_points.detach().cpu().numpy()
    matrices = intrinsics.detach().cpu().numpy()
    result = []
    for points, view in zip(xyz, matrices):
        points = points.T
        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view
        count = points.shape[1]
        points = np.concatenate((points, np.ones((1, count))))
        points = np.dot(viewpad, points)[:3, :]
        points = points / points[2:3, :].repeat(3, 0).reshape(3, count)
        result.append(points.T[:, :2])
    return torch.from_numpy(np.stack(result)).to(camera_points.device)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('pointaugmenting_project_camera_points_to_images', _impl)


def _meta(camera_points, intrinsics):
    """标准实浮点输入的原 NumPy 提升为 float64；固定六视图、N 不变。"""
    _check(camera_points, intrinsics)
    n = camera_points.shape[1]
    stride = (0, 0, 0) if n == 0 else (2, 2, 1) if n == 1 else (2 * n, 1, n)
    return torch.empty_strided((6, n, 2), stride, dtype=torch.float64, device='meta')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('pointaugmenting_project_camera_points_to_images', _meta)
register_no_grad('pointaugmenting_project_camera_points_to_images')


def pointaugmenting_project_camera_points_to_images(camera_points, intrinsics):
    """调用六相机像素投影算子；完整坐标、边界和 dtype 契约见 _impl。"""
    return torch.ops.fusionrepo.pointaugmenting_project_camera_points_to_images(camera_points, intrinsics)
