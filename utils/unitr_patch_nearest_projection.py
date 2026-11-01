"""UniTR patch-center nearest visible pseudo-grid projection."""

import torch
from torch.library import Library


_QUERY_CHUNK_SIZE = 128


@torch.no_grad()
def _impl(patch_coords, mapped_xyz, mapped_uv, counts, hw_shape, batch_size: int, num_views: int):
    patch_h, patch_w = hw_shape
    per_image = patch_coords[patch_coords[:, 0] == 0, 2:].clone().float()
    per_image[:, 0] = (per_image[:, 0] + 0.5) / patch_w
    per_image[:, 1] = (per_image[:, 1] + 0.5) / patch_h
    query = per_image.unsqueeze(0).repeat(batch_size * num_views, 1, 1).reshape(-1, 2)
    query3 = torch.cat((query, torch.zeros_like(query[:, :1])), -1).to(torch.float32)
    counts = counts.to(torch.int64)
    query_per_group = query.shape[0] // (batch_size * num_views)
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    nearest_global = torch.empty((query3.shape[0],), dtype=torch.long, device=query3.device)
    row = 0
    for group in range(batch_size * num_views):
        qn = query_per_group
        start = int(starts[group].item())
        count = int(counts[group].item())
        if count == 0:
            nearest_global[row:row + qn] = start
        else:
            known = mapped_uv[start:start + count].to(torch.float32)
            for query_start in range(row, row + qn, _QUERY_CHUNK_SIZE):
                query_end = min(query_start + _QUERY_CHUNK_SIZE, row + qn)
                unknown = query3[query_start:query_end]
                delta = unknown[:, None, :] - known[None, :, :]
                dist2 = (delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]) + delta[..., 2] * delta[..., 2]
                nearest_local = torch.argmin(dist2, dim=-1)
                nearest_global[query_start:query_end] = nearest_local + start
        row += qn
    idx = nearest_global[:, None].expand(-1, 3)
    nearest_xyz = torch.gather(mapped_xyz, 0, idx)
    neighbor_uv = torch.gather(mapped_uv, 0, idx)
    offset = (query3[:, :2] - neighbor_uv[:, :2]).abs()
    offset[:, 0] *= patch_w
    offset[:, 1] *= patch_h
    return nearest_xyz, offset


_LIB_IMPL = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_LIB_IMPL.impl("unitr_patch_nearest_projection", _impl)


def unitr_patch_nearest_projection(patch_coords, mapped_xyz, mapped_uv, counts, hw_shape, batch_size, num_views):
    """将每个 image patch 中心映射到同 batch/view 中最近的可见 pseudo-grid 点。

    Args:
        patch_coords: (P,4) 整数，列为 flattened view、z、两个 patch 网格轴；源码将第 3 列按 patchW
            解释为横向索引、第 4 列按 patchH 解释为纵向索引。所有张量须位于同一 device。只使用第 0
            view 的行生成查询，随后按 batch/view 数复制查询。
        mapped_xyz: (K,3) 浮点 LiDAR XYZ，单位米，按 batch/view 连续拼接。
        mapped_uv: (K,3) 有限浮点归一化图像坐标 (x,y,0)，顺序与 mapped_xyz 相同；NaN/Inf 不支持。
        counts: (batch_size*num_views,) 整数（调用者通常传 int32），每组 mapped 点数量，累计和须与 K 一致。
        hw_shape: (patchH,patchW) 正整数 patch 网格尺寸。
        batch_size, num_views: 正整数，分别是 batch 数和相机 view 数。

    Returns:
        nearest_xyz: (P0*batch_size*num_views,3)，从 mapped_xyz gather，保留其 dtype/device，单位米。
        nearest_offset: 同前缀 (P0*batch_size*num_views,2) float32 像素绝对偏移 (dx,dy)，位于输入 device。
        P0 是 patch_coords 中 flattened view 为 0 的行数，依赖坐标值而非张量 shape。

    以 float32 平方距离比较，严格同距时选组内首个点（argmin 首个最小值）；有限 UV 是输入前置条件，
    NaN 时其行为与源 stable argsort 不保证一致。距离与索引选择不可微。每组少于 3 个点及空组沿用
    原始 three_nn 首索引回退；若最后一个空组回退到 K，则后续 gather 会越界。数据依赖的输出行数无准确
    Meta kernel。来源：UniTR `unitr_utils.py:145-172`、`pointnet2_utils.py:228-261`、
    `interpolate_gpu.cu:16-75`。
    """
    return torch.ops.fusionrepo.unitr_patch_nearest_projection(
        patch_coords, mapped_xyz, mapped_uv, counts, list(hw_shape), batch_size, num_views
    )
