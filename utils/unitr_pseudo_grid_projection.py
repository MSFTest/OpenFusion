"""UniTR pseudo-LiDAR grid projection and visibility compaction.

Projection remains the source tensor formula: the normalized-grid helper can round a tiny positive raw pixel to
the -1 boundary before visibility selection, and ``points_to_image`` differs from the source by float32 ULPs on
random projections even with eps=1e-5. Both alter strict boundary visibility or downstream coordinates.
"""

import torch
from torch.library import Library


@torch.no_grad()
def _impl(points, lidar2image, image_aug_matrix, lidar2image_aug, image_shape, training: bool, use_map: bool):
    h, w = image_shape
    points = points.to(torch.float32)
    if training and lidar2image_aug is not None and not use_map:
        lidar2image = lidar2image_aug
    lidar2image = lidar2image.to(torch.float32)
    image_aug_matrix = image_aug_matrix.to(torch.float32)
    batch_size, views = lidar2image.shape[:2]
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    projected = torch.einsum("bvij,gsj->bvgsi", lidar2image, homogeneous)
    mask = projected[..., 2:3] > 1e-5
    projected[..., :2] = projected[..., :2] / torch.maximum(
        projected[..., 2:3], torch.ones_like(projected[..., 2:3]) * 1e-5
    )
    projected[..., 2] = 1
    if image_aug_matrix.shape[1] == 1 and views != 1:
        image_aug_matrix = image_aug_matrix.expand(-1, views, -1, -1)
    uv = torch.einsum("bvij,bvgsj->bvgsi", image_aug_matrix, projected)[..., :2]
    uv[..., 0] /= w
    uv[..., 1] /= h
    mask = mask & (uv[..., 1:2] > 0) & (uv[..., 1:2] < 1) & (uv[..., 0:1] > 0) & (uv[..., 0:1] < 1)
    mask = torch.nan_to_num(mask).squeeze(-1)
    counts = mask.reshape(batch_size, views, -1).sum(-1).reshape(-1).int()
    mapped_uv_valid = uv[mask]
    mapped_uv = torch.cat((mapped_uv_valid, torch.zeros_like(mapped_uv_valid[:, :1])), dim=-1)
    xyz = points[None, None].expand(batch_size, views, -1, -1, -1)[mask]
    return xyz, mapped_uv, counts


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("unitr_pseudo_grid_projection", _impl)


def unitr_pseudo_grid_projection(
    points, lidar2image, image_aug_matrix, image_shape, lidar2image_aug=None, training=False, use_map=False
):
    """将全局 pseudo-LiDAR 网格投影至各 batch/view 图像并压缩保留可见采样点。

    Args:
        points: (G,S,3) 数值当前 LiDAR XYZ，单位米；内部转 float32，允许 G 或 S 为 0。
        lidar2image: (B,V,4,4) 浮点齐次矩阵，从当前 LiDAR 坐标映射至相机投影坐标。
        image_aug_matrix: (B,V,4,4) 或 (B,1,4,4) 浮点图像增强矩阵。
        image_shape: (H,W) 图像高宽，单位像素。
        lidar2image_aug: 可选 (B,V,4,4) 增强后 LiDAR 投影矩阵。
        training, use_map: 仅 training=True、增强矩阵存在且 use_map=False 时选择 lidar2image_aug。

    Returns:
        mapped_xyz: (K,3) float32，当前 LiDAR XYZ，单位米，按 batch 再按 view 顺序压缩，位于输入 device。
        mapped_uv: (K,3) float32，图像归一化坐标 (x,y,0)，每个坐标属于 (0,1)，顺序与 mapped_xyz 一致。
        counts: (B*V,) int32，每个 batch/view 的可见采样数，位于输入 device；K 为 counts.sum()。
        points 与矩阵必须位于同一 device；计算只接受浮点 points，内部统一投影 dtype 为 float32。

    保留 UniTR 的齐次矩阵顺序与原始归一化图像坐标计算，避免先转换为 [-1,1] grid 再转回时在边界
    附近量化掉微小正坐标。深度要求 >1e-5，图像边界严格为 (0,1)。空可见集合返回 K=0 和零 counts。
    投影/可见性压缩不可微；K 依赖运行时可见性，无准确 Meta kernel。来源：UniTR
    `unitr_utils.py:40-89,111-143`。
    """
    return torch.ops.fusionrepo.unitr_pseudo_grid_projection(
        points, lidar2image, image_aug_matrix, lidar2image_aug, list(image_shape), training, use_map
    )
