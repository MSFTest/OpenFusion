import numpy as np
import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils._pointaugmenting_dispatch import check_numpy_float, register_no_grad


def _check(points, ref_to_global, cameras_from_global):
    check_numpy_float(points, ref_to_global, cameras_from_global)
    if points.ndim != 2 or points.shape[1] != 3 or ref_to_global.shape != (4, 4) or cameras_from_global.shape != (6, 4, 4):
        raise ValueError('需要 points[N,3]、ref_to_global[4,4]、cameras_from_global[6,4,4]')


def _impl(points, ref_to_global, cameras_from_global):
    """将增强前参考 LiDAR 点经 global 变换到六个相机，保留原 NumPy 顺序。

    来源：原 det3d/datasets/pipelines/loading.py::LoadPointCloudFromFile.__call__
    190–196 行；先 ref_to_global.dot，再 camera_from_global.dot，保留中间齐次 w。
    Args:
        points: 实浮点 [N,3]，参考 LiDAR xyz，单位米；允许空点和非连续布局。
        ref_to_global: 实浮点 [4,4]，参考 LiDAR→global，左乘列向量。
        cameras_from_global: 实浮点 [6,4,4]，global→六相机；顺序由调用者保留。
        支持 float16/32/64 及混合 dtype/device；各输入显式复制到 CPU 计算。
    Returns:
        float64 [6,N,3] 相机 xyz，单位米，z 为相机深度；设备跟随 points。
        保留 NumPy stack 布局：N>1 时 stride=(3*N,1,N)，N=0/1 使用原退化布局。
        原 float64 ones 的提升不改变；不合并矩阵、不筛负深度、不修改输入。
        无参数或状态，不可微；不修复非有限值，支持一般 4×4 矩阵。
    """
    _check(points, ref_to_global, cameras_from_global)
    xyz = points.detach().cpu().numpy()
    ref = ref_to_global.detach().cpu().numpy()
    cameras = cameras_from_global.detach().cpu().numpy()
    result = []
    for camera in cameras:
        homogeneous = np.concatenate([xyz[:, :3], np.ones([xyz.shape[0], 1])], axis=1)
        global_points = ref.dot(homogeneous.T)
        camera_points = camera.dot(global_points)[:3, :]
        result.append(camera_points.T)
    return torch.from_numpy(np.stack(result)).to(points.device)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('pointaugmenting_transform_reference_points_to_cameras', _impl)


def _meta(points, ref_to_global, cameras_from_global):
    """原实浮点和 float64 ones 的提升固定为 float64；N 不依赖点值。"""
    _check(points, ref_to_global, cameras_from_global)
    n = points.shape[0]
    stride = (0, 0, 0) if n == 0 else (3, 3, 1) if n == 1 else (3 * n, 1, n)
    return torch.empty_strided((6, n, 3), stride, dtype=torch.float64, device='meta')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('pointaugmenting_transform_reference_points_to_cameras', _meta)
register_no_grad('pointaugmenting_transform_reference_points_to_cameras')


def pointaugmenting_transform_reference_points_to_cameras(points, ref_to_global, cameras_from_global):
    """调用参考 LiDAR→global→六相机算子；完整输入输出契约见 _impl。"""
    return torch.ops.fusionrepo.pointaugmenting_transform_reference_points_to_cameras(
        points, ref_to_global, cameras_from_global)
