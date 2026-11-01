import torch
from torch.library import Library
from utils import operator_registry as _registry


def _impl(points, angle):
    '原CenterPoint box_torch_ops.rotate_points_along_z行向量旋转。\nArgs:\n    points: 同设备float32[B,N,3+C]，前3列为米制xyz；angle: float32[B]弧度。\nReturns:\n    float32[B,N,3+C]，同设备，可微；正角按原行向量公式顺时针，附加C列原样拼接。\n    支持空N与非contiguous输入；不会把速度等附加列旋转。B需与angle一致。\n'
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((cosa, -sina, zeros, sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot

_lib_impl=Library('fusionrepo','IMPL','CompositeExplicitAutograd')
_lib_impl.impl('mvp_rotate_points_along_z',_impl)
_lib_autograd=Library('fusionrepo','IMPL','Autograd')
_lib_autograd.impl('mvp_rotate_points_along_z',_impl)

def _meta(points,angle):
    if points.dtype!=torch.float32:raise RuntimeError('original rotation matrix is float32')
    return points.new_empty(points.shape)
_lib_meta=Library('fusionrepo','IMPL','Meta')
_lib_meta.impl('mvp_rotate_points_along_z',_meta)

def mvp_rotate_points_along_z(points, angle):
    """调用 fusionrepo::mvp_rotate_points_along_z；shape/dtype/device/边界及坐标见 _impl。"""
    return torch.ops.fusionrepo.mvp_rotate_points_along_z(points,angle)
