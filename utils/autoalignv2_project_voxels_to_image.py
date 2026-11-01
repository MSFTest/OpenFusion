import torch
from torch.library import Library

from utils import operator_registry as _registry
from utils.batched_homogeneous_transform_and_normalize import batched_homogeneous_transform_and_normalize


def _impl(centers, projection, scale, crop):
    """将原 LiDAR XYZ 米制点投影为 resize/crop 后的像素及投影深度。

    来源：multi_voxel_deformfusion.py::point_sample 61–70 行及 points_cam2img。
    Args:
        centers: 浮点 [M,3]；projection: 同 dtype/device 的 [3,3]/[3,4]/[4,4]
            LiDAR→图像齐次投影，行向量计算等价于 [x,y,z,1] @ projection.T。
        scale/crop: 同 device 浮点标量或 [2]，依次对像素 uv 缩放/减偏移。
    Returns:
        同 device [M,3]，列为像素 u/v/原投影深度。输出 dtype 服从原算术提升。
        不裁剪、不筛选、不翻转图像；深度除法无 epsilon，0/负深度保持 inf/NaN
        或负值。允许空/非连续输入；浮点路径可微。无参数、buffer 或训练状态。
    """
    assert projection.ndim == 2
    rows, cols = projection.shape
    assert (rows, cols) in ((3, 3), (3, 4), (4, 4))
    if rows == 3:
        expanded = torch.eye(4, dtype=projection.dtype, device=projection.device)
        expanded[:rows, :cols] = projection
        projection = expanded
    projected = batched_homogeneous_transform_and_normalize(centers, projection, False).squeeze(0)
    uv = projected[:, :2] / projected[:, 2:3]
    uv = uv * scale
    uv -= crop
    return torch.cat([uv, projected[:, 2:3]], -1)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('autoalignv2_project_voxels_to_image', _impl)


def _meta(centers, projection, scale, crop):
    return _impl(centers, projection, scale, crop)


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('autoalignv2_project_voxels_to_image', _meta)


def autoalignv2_project_voxels_to_image(centers, projection, scale, crop):
    """调用 AutoAlignV2 uvd 投影；矩阵方向和边界见 _impl。"""
    return torch.ops.fusionrepo.autoalignv2_project_voxels_to_image(centers, projection, scale, crop)
