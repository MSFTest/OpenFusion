import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


@torch.no_grad()
def _impl(pixels, cameras, depths, indices, point_labels, intrinsics, transforms):
    """原 nusc_image_projection.py:38–51、virtual_gen.py:132–161 的语义虚拟点。

    Args:
        pixels: float32 [Q,3]，原图 u,v,编码实例编号；第三列不参与反投影。
        cameras: float32 [Q,1]，编号必须落在 [0,C)，每点恰属一相机。
        depths: float32 [Q] 相机 Z（米）；indices: int64 [Q] 相机主序 CN 索引。
        point_labels: float32 [CN,11]；intrinsics: float32 [C,3,3] 完整 K。
        transforms: float32 [C,4,4] LiDAR→camera T；输入须同设备。
    Returns:
        同设备 float32 [Q,14]，LiDAR 米制 XYZ 加十一维语义，按相机分组且组内稳定。
        labels 非正式提升 dtype 时，输出遵循原 cat 类型提升；不可微。
    原逆 K、逆 T 分别执行，不提前合并。空相机/总 Q=0 支持；奇异矩阵抛错。
    Q=1 保留源码标量 squeeze 导致的 RuntimeError；本次格式重构不扩展该边界。
    Meta 仅支持合法 camera_id 契约；不读取相机数据，Q=1 同样明确拒绝。
    """
    all_points, all_labels = [], []
    for camera in range(len(intrinsics)):
        camera_mask = (cameras == camera).squeeze()
        selected_pixels = pixels[camera_mask]
        selected_indices = indices[camera_mask]
        selected_depth = depths[camera_mask].reshape(1, -1)
        selected_pixels = selected_pixels[:, :2]
        padded = torch.cat([selected_pixels.transpose(1, 0).float(),
                            torch.ones(1, len(selected_pixels), device=indices.device)], dim=0)
        count = padded.shape[1]
        scaled = padded * selected_depth.repeat(3, 1).reshape(3, count)
        inverse_k = torch.eye(4, device=pixels.device)
        inverse_k[:intrinsics.shape[1], :intrinsics.shape[2]] = torch.inverse(intrinsics[camera])
        # 保留原二维 matmul 及非连续布局；通用 batch1 affine 的舍入不完全相同。
        scaled = torch.cat([scaled, torch.ones(1, count, device=pixels.device)], dim=0)
        xyz_camera = torch.matmul(inverse_k, scaled)[:3]
        xyz_camera[:3] = torch.matmul(torch.inverse(transforms[camera]), torch.cat([
            xyz_camera[:3], torch.ones(1, xyz_camera.shape[1], device=indices.device)
        ], dim=0))[:3]
        all_points.append(xyz_camera.transpose(1, 0))
        all_labels.append(point_labels[selected_indices])
    return feature_concatenation([torch.cat(all_points, dim=0), torch.cat(all_labels, dim=0)], 1)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_unproject_semantic_virtual_points', _impl)


def _meta(pixels, cameras, depths, indices, point_labels, intrinsics, transforms):
    """每点一个合法 camera_id 时输出 Q×14；原 Q=1 异常不能伪报成功。"""
    if pixels.shape[0] == 1:
        raise RuntimeError('原标量 squeeze 使 Q=1 反投影报错')
    return torch.empty((pixels.shape[0], 14), dtype=torch.promote_types(torch.float32, point_labels.dtype), device='meta')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('mvp_unproject_semantic_virtual_points', _meta)
register_inference_autograd('mvp_unproject_semantic_virtual_points')


def mvp_unproject_semantic_virtual_points(pixels, cameras, depths, indices, point_labels, intrinsics, transforms):
    """同设备像素、米制 Z、CN 索引、语义及 K/T→相机分组的 LiDAR [Q,14]。

    完整矩阵方向、输入 dtype、Q=1 限制及不可微契约见 _impl。
    """
    return torch.ops.fusionrepo.mvp_unproject_semantic_virtual_points(pixels, cameras, depths, indices, point_labels, intrinsics, transforms)
