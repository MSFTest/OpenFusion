"""恢复原 LiDAR 米制坐标，源 coord_transform.py:7 reverse=True 及 LiDARPoints。
Args: points[N,3]、rotation[3,3] 右乘旋转矩阵，scale 标量，translation[3]或[N,3]；
hflip/vflip 为 y/x 反号开关，flow 按实际增强执行顺序列出 T/S/R/HF/VF。
Returns: 新 float32[N,3]，device 跟随 points，允许非连续/空点；逆序执行 flow，
先计算全部逆参数，零 scale 和奇异矩阵报错，即使 flow 为空也不跳过参数准备。
源 Tensor 转 float32 语义保持；输入应同 device，原几何调用者会 detach 输出。
Meta/Fake shape 精确，但奇异性是数据依赖错误，只能 eager 判定。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry


def _impl(points, rotation, scale, translation, hflip, vflip, flow):
    points = torch.as_tensor(points.clone(),dtype=torch.float32,device=points.device)
    # coord_transform.py:65–71在执行flow前即构造全部逆变换参数；保留奇异矩阵/零scale边界。
    inverse_scale = 1.0/scale
    inverse_translation = -translation
    inverse_rotation = rotation.inverse()
    for op in reversed(flow):
        assert op in ('T','S','R','HF','VF'), f'This 3D data transformation op ({op}) is not supported'
        if op == 'T':
            trans = inverse_translation.squeeze(0)
            if trans.dim() == 1:
                assert trans.shape[0] == 3
            elif trans.dim() == 2:
                assert trans.shape[0] == points.shape[0] and trans.shape[1] == 3
            else:
                raise NotImplementedError('Unsupported translation vector shape')
            points[:,:3] += trans
        elif op == 'S':
            points[:,:3] *= inverse_scale
        elif op == 'R':
            points[:,:3] = points[:,:3] @ inverse_rotation
        elif op == 'HF' and hflip:
            points[:,1] = -points[:,1]
        elif op == 'VF' and vflip:
            points[:,0] = -points[:,0]
    return points[:,:3]


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transfusion_reverse_lidar_transform", _impl)


def _meta(points, rotation, scale, translation, hflip, vflip, flow):
    if scale == 0:
        raise ZeroDivisionError("float division by zero")
    rotation.inverse()  # 检查可静态判定的inverse shape/dtype；不读取奇异性数据。
    for tag in flow:
        assert tag in ("T", "S", "R", "HF", "VF")
    if 'R' in flow and rotation.dtype != torch.float32:
        raise RuntimeError("原 float32 点无法右乘其他 dtype 旋转矩阵")
    return torch.empty((points.shape[0], 3), dtype=torch.float32, device="meta")


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transfusion_reverse_lidar_transform", _meta)


def transfusion_reverse_lidar_transform(points, rotation, scale, translation, hflip, vflip, flow):
    """恢复原 LiDAR 米制坐标，源 coord_transform.py:7 reverse=True 及 LiDARPoints。
Args: points[N,3]、rotation[3,3] 右乘旋转矩阵，scale 标量，translation[3]或[N,3]；
hflip/vflip 为 y/x 反号开关，flow 按实际增强执行顺序列出 T/S/R/HF/VF。
Returns: 新 float32[N,3]，device 跟随 points，允许非连续/空点；逆序执行 flow，
先计算全部逆参数，零 scale 和奇异矩阵报错，即使 flow 为空也不跳过参数准备。
源 Tensor 转 float32 语义保持；输入应同 device，原几何调用者会 detach 输出。
Meta/Fake shape 精确，但奇异性是数据依赖错误，只能 eager 判定。
"""
    return torch.ops.fusionrepo.transfusion_reverse_lidar_transform(points, rotation, scale, translation, hflip, vflip, flow)
