import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils.transfusion_reverse_lidar_transform import transfusion_reverse_lidar_transform


def _impl(centers, rotation, scale, translation, horizontal, vertical, flow):
    """逆序还原 LiDAR 增强，复用既有逆变换并保留原项目异常政策。

    来源：coord_transform.py::apply_3d_transformation(reverse=True)，及
    BasePoints/LiDARPoints；调用者为 multi_voxel_deformfusion.py::point_sample。
    Args:
        centers: 浮点 [M,3] XYZ 米制点，允许空/非连续输入。
        rotation: 同 device [3,3] 右乘旋转矩阵；scale: Python 浮点缩放。
        translation: 同 device [3] 米制位移。
        horizontal/vertical: LiDAR y/x 轴翻转标志。
        flow: 正向增强的 T/S/R/HF/VF 顺序，逆序执行。
    Returns:
        同 device float32 [M,3] 原 LiDAR 系坐标，不修改输入点。
        点坐标可微，metadata 由调用者复制为常量。零 scale 即使 flow 为空也
        报错；旋转求逆失败重试一次，再失败按原源码打印并使用单位旋转。
        无 box 操作、参数、buffer 或 train/eval 状态。
    """
    1.0 / scale  # 原源码在准备旋转前就准备逆 scale，不能由 flow 是否含 S 决定。
    safe_rotation = rotation
    try:
        rotation.inverse()
    except RuntimeError:
        print('Rotate Error')
        print(rotation)
        try:
            rotation.inverse()
        except Exception:
            print('Double Err')
            safe_rotation = torch.eye(3, dtype=centers.dtype, device=centers.device)
    return transfusion_reverse_lidar_transform(
        centers, safe_rotation, scale, translation, horizontal, vertical, flow)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_restore_lidar_coordinates', _impl)


def _meta(centers, rotation, scale, translation, horizontal, vertical, flow):
    # 奇异性仅在 eager 可判定；已有 helper 的 Meta 不读矩阵值。
    return transfusion_reverse_lidar_transform(
        centers, rotation, scale, translation, horizontal, vertical, flow)


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_restore_lidar_coordinates', _meta)


def autoalignv2_restore_lidar_coordinates(centers, rotation, scale, translation, horizontal, vertical, flow):
    """调用 AutoAlignV2 逆增强；XYZ/米、异常及梯度契约见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_restore_lidar_coordinates(
        centers, rotation, scale, translation, horizontal, vertical, flow)
