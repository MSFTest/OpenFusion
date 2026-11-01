import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd
from utils.projected_pixel_valid_mask import projected_pixel_valid_mask


@torch.no_grad()
def _impl(points, transforms, intrinsics, height=900, width=1600):
    """原 nusc_image_projection.py:16–36、59–92 及 virtual_gen.py:230–232。

    Args:
        points: float32 [N,4] LiDAR xyzr，XYZ 单位米，支持非连续布局。
        transforms: 同设备 float32 [C,4,4]，LiDAR→camera 齐次外参。
        intrinsics: 同设备 float32 [C,3,3]，完整相机内参，不限定零 skew。
        height,width: 图像高宽像素，正式默认 900、1600。
    Returns:
        输入设备 float32 [C,N,5]，相机主序 [u,v,camera_z,valid,camera_id]。
        u/v 经过 floor，深度为相机 Z（米），valid 为 0/1；不可微。
    边界：严格排除 u/v=0、越界、非正深度；无效行前四列置零。
    无 epsilon 或深度 clamp；保留 4×4 bmm 及除法顺序，避免改变整数像素。
    N=0 支持。混合 dtype/device 沿用矩阵乘法异常，不静默迁移。
    """
    count, cameras = points.shape[0], len(transforms)
    projected = torch.zeros((cameras, count, 4), device=points.device, dtype=torch.float32)
    # 通用 affine 在非连续输入下的 GEMM 舍入与原 einsum 有差异；这里紧邻 floor，
    # 必须保留原乘法形状/顺序，不能把几何近似相等当成整数索引等价。
    padded_points = torch.cat([points.transpose(1, 0)[:3],
                               torch.ones(1, count, dtype=points.dtype, device=points.device)], dim=0)
    camera_points = torch.einsum('abc,cd->abd', transforms, padded_points)[:, :3]
    depths = camera_points[:, 2]
    viewpad = torch.eye(4, device=points.device, dtype=torch.float32).unsqueeze(0).repeat(cameras, 1, 1)
    viewpad[:, :intrinsics.shape[1], :intrinsics.shape[2]] = intrinsics
    padded = torch.cat([camera_points, torch.ones(cameras, 1, count, device=points.device)], dim=1)
    image_points = torch.bmm(viewpad, padded)[:, :3]
    image_points = image_points / image_points[:, 2:3].repeat(1, 3, 1)
    pixels = torch.floor(image_points[:, :2].transpose(2, 1))
    x, y = pixels[..., 0].long(), pixels[..., 1].long()
    valid = projected_pixel_valid_mask(torch.stack([x, y], -1), depths[..., None], height, width, 0.)
    valid &= (x > 0) & (y > 0)
    selected = projected[valid]
    selected[:, :2] = pixels[valid]
    selected[:, 2] = depths[valid]
    selected[:, 3] = 1
    projected[valid] = selected
    camera_ids = torch.arange(cameras, dtype=torch.float32, device=points.device).reshape(cameras, 1, 1).repeat(1, count, 1)
    return torch.cat([projected, camera_ids], dim=-1)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_project_lidar_to_images', _impl)


def _meta(points, transforms, intrinsics, height=900, width=1600):
    """合法 float32 输入的固定 C×N×5 输出；不读取 Tensor 数据。"""
    if points.dtype != transforms.dtype or points.dtype != torch.float32:
        raise RuntimeError('原投影要求 points/transforms 为 float32')
    return torch.empty((transforms.shape[0], points.shape[0], 5), dtype=torch.float32, device='meta')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('mvp_project_lidar_to_images', _meta)
register_inference_autograd('mvp_project_lidar_to_images')


def mvp_project_lidar_to_images(points, transforms, intrinsics, height=900, width=1600):
    """LiDAR float32 [N,4] 经 T[C,4,4]、K[C,3,3]→[C,N,5]。

    原图像素、米制相机 Z、严格零边界、无效归零及无梯度语义见 _impl。
    """
    return torch.ops.fusionrepo.mvp_project_lidar_to_images(points, transforms, intrinsics, height, width)
