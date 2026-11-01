"""UniTR LiDAR pillar sample projection into image patch coordinates.

Projection remains the source tensor formula: the normalized-grid helper can round a tiny positive raw pixel to
the -1 boundary before visibility selection, and ``points_to_image`` differs from the source by float32 ULPs on
random projections even with eps=1e-5. Both alter strict boundary visibility or downstream coordinates.
"""

import torch
from torch.library import Library


@torch.no_grad()
def _impl(points, batch_idx, lidar2image, image_aug_matrix, lidar2image_aug, image_shape, hw_shape,
          training: bool, use_map: bool, patch_coords, num_views: int):
    h, w = image_shape
    patch_h, patch_w = hw_shape
    points = points.to(torch.float32)
    lidar2image = (lidar2image_aug if training and lidar2image_aug is not None and not use_map else lidar2image).to(torch.float32)
    image_aug_matrix = image_aug_matrix.to(torch.float32)
    out = []
    # Like MapLidar2Image, the source uses the final batch_idx to determine iteration range.
    batch_size = int((batch_idx[-1] + 1).item())
    for b in range(batch_size):
        pts = points[batch_idx == b]
        homogeneous = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
        projected = torch.matmul(lidar2image[b:b + 1, :, None, None], homogeneous[None, None, ..., None]).squeeze(-1)
        mask = projected[..., 2:3] > 1e-5
        projected[..., :2] = projected[..., :2] / torch.maximum(
            projected[..., 2:3], torch.full_like(projected[..., 2:3], 1e-5)
        )
        projected[..., 2] = 1
        uv = torch.matmul(image_aug_matrix[b:b + 1, :, None, None], projected[..., None]).squeeze(-1)[..., :2]
        uv[..., 0] /= w
        uv[..., 1] /= h
        mask = mask & (uv[..., 1:2] > 0) & (uv[..., 1:2] < 1) & (uv[..., 0:1] > 0) & (uv[..., 0:1] < 1)
        mask = torch.nan_to_num(mask).squeeze(-1).squeeze(3).permute(0, 2, 1).reshape(-1, lidar2image.shape[1])
        hit = mask.any(-1)
        mask[~hit, 0] = True
        ids = torch.nonzero(mask)
        first = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        first[1:] = ids[1:, 0] != ids[:-1, 0]
        chosen = ids[first]
        uv = uv.squeeze(3).permute(0, 2, 1, 3).flatten(0, 1)
        selected = uv[torch.arange(uv.shape[0], device=uv.device), chosen[:, 1]]
        selected = torch.clamp(selected, -1, 2) + 1
        view = chosen[:, 1:2].to(selected.dtype)
        out.append(torch.cat((view, selected[:, 0:1] * patch_w, selected[:, 1:2] * patch_h), dim=-1))
    lidar_projected = torch.cat(out, dim=0)
    lidar_joint_coords = torch.cat((batch_idx.reshape(-1, 1), lidar_projected), dim=1)
    image_joint_coords = patch_coords.clone()
    image_joint_coords[:, 0] = patch_coords[:, 0] // num_views
    image_joint_coords[:, 1] = patch_coords[:, 0] % num_views
    image_joint_coords[:, 2] += patch_w
    image_joint_coords[:, 3] += patch_h
    return lidar_joint_coords, image_joint_coords


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("unitr_lidar_to_image_projection", _impl)


def _meta(points, batch_idx, lidar2image, image_aug_matrix, lidar2image_aug, image_shape, hw_shape,
          training: bool, use_map: bool, patch_coords, num_views: int):
    return (
        points.new_empty((points.shape[0], 4), dtype=torch.float32),
        patch_coords.new_empty(patch_coords.shape),
    )


_LIB_META = Library("fusionrepo", "IMPL", "Meta")
_LIB_META.impl("unitr_lidar_to_image_projection", _meta)


def unitr_lidar_to_image_projection(
    points, batch_idx, lidar2image, image_aug_matrix, image_shape, hw_shape, patch_coords, num_views,
    lidar2image_aug=None, training=False, use_map=False,
):
    """将 LiDAR 柱体采样点投影到首个命中相机的 patch 网格，并生成图像 patch 的联合分窗坐标。

    Args:
        points: (N,1,3) 数值增强后 LiDAR XYZ，单位米；内部转 float32，源实现只支持每柱一个采样点。
        batch_idx: (N,) 整数 batch id，与 points 同 device；要求非空、按 batch 连续排列且最后值是最大 batch id。
        lidar2image: (B,V,4,4) 浮点齐次矩阵，将当前 LiDAR 坐标映射到相机投影坐标。
        image_aug_matrix: (B,V,4,4) 或 (B,1,4,4) 浮点图像增强齐次矩阵。
        image_shape: (H,W) 原图高宽，单位像素。
        hw_shape: (patchH,patchW) patch 网格尺寸。
        patch_coords: (P,4) 整数 flattened-view patch 坐标，列为 (batch*V,z,patch_y,patch_x)。
        num_views: 相机 view 数 V。
        lidar2image_aug: 可选 (B,V,4,4) 增强后 LiDAR 投影矩阵。
        training, use_map: 仅 training=True、增强矩阵存在且 use_map=False 时选择 lidar2image_aug。

    Returns:
        lidar_joint_coords: (N,4) float32，列值为 (batch,view,projected_x*patchW,projected_y*patchH)，
        按下游 UniTR sparse 坐标约定作为 (batch,z,y,x) 使用。
        image_joint_coords: (P,4)，复制 patch_coords 的 dtype/device，将 flattened view 拆为 batch/view，
        并对第 3、4 列分别加 patchW、patchH 偏移。
        两输出保持各自输入 device；points 与矩阵必须同 device。

    深度阈值为 1e-5，增强后归一化图像坐标的可见边界严格是 (0,1)。多 view 取首个有效 view；无命中
    回退到 view 0。送入分窗前归一化坐标 clamp 到 [-1,2] 后加 1。成功输入时 LiDAR 输出为 (N,4)，
    图像输出与 patch_coords 同 shape，二者的 Meta shape 均可由输入 shape 静态推导。空 batch_idx 沿用源 IndexError；S>1 沿用 squeeze(3) RuntimeError。
    坐标构造不可微。严格边界回归来自 `points=[[[1e-7,20,1]]]` 在 image_shape=(40,40)：归一化 grid core
    把 x 映为 -1 并判不可见，源公式则保留正像素坐标。来源：UniTR `unitr_utils.py:40-89,239-282`、
    `mm_backbone/unitr.py:306-320`。
    """
    return torch.ops.fusionrepo.unitr_lidar_to_image_projection(
        points, batch_idx, lidar2image, image_aug_matrix, lidar2image_aug, list(image_shape), list(hw_shape),
        training, use_map, patch_coords, num_views,
    )
