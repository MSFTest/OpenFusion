import torch
from torch.nn import functional as F
from torch.library import Library

from utils import operator_registry as _registry
from utils._pointaugmenting_dispatch import register_no_grad
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


def _check(image_features, voxels, sample_grid, voxel_valid):
    if image_features.ndim != 5 or image_features.shape[1:3] != (6, 64):
        raise ValueError('需要 image_features[B,6,64,H,W]')
    if voxels.ndim != 3 or voxels.shape[2] != 9 or voxels.shape[1] <= 0:
        raise ValueError('需要 voxels[Nv,T,9]，T>0；原 voxel/pp 配置 T=10/20')
    if sample_grid.ndim != 5 or sample_grid.shape[0] != image_features.shape[0] or sample_grid.shape[2:] != (voxels.shape[1], 1, 3):
        raise ValueError('需要 sample_grid[B,M,T,1,3]')
    if voxel_valid.dtype != torch.bool or voxel_valid.shape != (sample_grid.shape[0] * sample_grid.shape[1],):
        raise ValueError('需要 bool voxel_valid[B*M]')
    if voxels.shape[0] > voxel_valid.numel():
        raise ValueError('Nv 不得大于 voxel_valid 的 B*M 槽位数')
    if image_features.dtype not in (torch.float32, torch.float64) or sample_grid.dtype != image_features.dtype:
        raise ValueError('image_features 和 sample_grid 需要相同 float32/float64 dtype')
    if voxels.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError('voxels 需要实浮点特征')
    if any(x.device != image_features.device for x in (voxels, sample_grid, voxel_valid)):
        raise RuntimeError('采样特征、体素、grid 和 voxel_valid 必须在相同 device')
    if image_features.shape[0] <= 0 or min(image_features.shape[-2:]) <= 0:
        raise ValueError('图像特征 B/H/W 必须为正数')


def _impl(image_features, voxels, sample_grid, voxel_valid):
    """按原五维采样、筛选与槽位 mask 顺序，将相机特征装饰到 LiDAR 点。

    来源：原 pp_fusion.py::get_img_feat/forward、voxelnet_fusion.py::get_img_feat/forward。
    Args:
        image_features: float32/64 [B,6,64,H,W]，外部图像模型已给出的特征；不构建模型。
        voxels: 实浮点 [Nv,T,9]，列为增强后参考 LiDAR xyzrt（米/原强度/秒）、
            绑定的归一化 u/v/camera、槽位 valid；正式 voxel/pp 的 T 为 10/20。
        sample_grid: 与 image_features 同 dtype 的 [B,M,T,1,3]，末维顺序 u/v/camera。
        voxel_valid: bool [B*M]，batch 主序的有效体素；合法域 sum(valid)==Nv。
        输入必须同 device；保持原 view/transpose/contiguous 顺序，不修复非法布局。
    Returns:
        实浮点 [Nv,T,69]，xyzrt 后接 64 通道相机特征；dtype 为特征与 voxels 提升结果，
        device 不变，不可微。空 Nv 允许；不合并重复槽位或重复像素，按原索引顺序筛选。
        相机作为深度轴；bilinear（五维即三线性）、zeros padding、原默认 align_corners。
        域外坐标沿用 grid_sample 语义；先采样/选择再乘槽位 valid，再裁去四列辅助字段。
        无学习参数或状态；不改变相机轴端点的半权重，不按 camera_id 直接 gather。
    """
    _check(image_features, voxels, sample_grid, voxel_valid)
    batch_size = image_features.shape[0]
    max_points = voxels.shape[1]
    image_features = image_features.view(batch_size, 6, -1, image_features.shape[3], image_features.shape[4]).transpose(2, 1)
    sampled = F.grid_sample(image_features, sample_grid, mode='bilinear', padding_mode='zeros')
    sampled = sampled.transpose(1, 4).contiguous()
    sampled = sampled.view(-1, max_points, 64).contiguous()
    sampled = sampled[voxel_valid]
    if sampled.shape[0] != voxels.shape[0]:
        raise ValueError('voxel_valid 选出的体素数量必须等于 voxels 的 Nv，不允许广播')
    sampled = sampled * voxels[:, :, -1].view(-1, max_points, 1)
    return feature_concatenation([voxels[:, :, :-4], sampled], 2)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('pointaugmenting_sample_and_decorate_point_features', _impl)


def _meta(image_features, voxels, sample_grid, voxel_valid):
    """合法域 sum(valid)==Nv 的输出由 voxels 精确确定，Meta 不读取 mask 值。

    mask 数值约束只能由 eager 检查；此处不是任意布尔选择的动态长度/上界声明。
    """
    _check(image_features, voxels, sample_grid, voxel_valid)
    dtype = torch.promote_types(image_features.dtype, voxels.dtype)
    return torch.empty((voxels.shape[0], voxels.shape[1], 69), dtype=dtype, device='meta')


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('pointaugmenting_sample_and_decorate_point_features', _meta)
register_no_grad('pointaugmenting_sample_and_decorate_point_features')


def pointaugmenting_sample_and_decorate_point_features(image_features, voxels, sample_grid, voxel_valid):
    """调用五维图像采样及点装饰；输入布局、合法域与边界见 _impl。"""
    return torch.ops.fusionrepo.pointaugmenting_sample_and_decorate_point_features(
        image_features, voxels, sample_grid, voxel_valid)
