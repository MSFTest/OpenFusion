"""相机独有体素指派 LiDAR 参考体素。原源码：
mmdet3d/models/middle_encoders/sparse_multimodal_encoder_painting.py，
SparseMultiModalEncoderPaint.pad_missing_batch_id 208–225、fps_NN_fast 276–323、
grouped_sparse_conv 337–369 行；mmdet3d/ops/furthest_point_sample/src/
furthest_point_sample_cuda.cu 25–140 行；mmdet3d/ops/ball_query/src/
ball_query_cuda.cu 11–54 行。
"""

from __future__ import annotations

import torch
from torch.library import Library


def _fps_reference(points: torch.Tensor, count: int) -> torch.Tensor:
    """CUDA FPS 参考：初始 0，逐点累计最小平方距离。

    原 kernel 每线程按 tid+stride*k 扫描，严格大于时更新；归约在
    相等时保留较低 tid。这一规则在相同最大距离时也按线程号处理。
    """
    n = points.shape[0]
    threads = min(1 << (n.bit_length() - 1), 1024)
    temp = torch.full((n,), 1e10, dtype=torch.float32, device=points.device)
    result = torch.zeros(count, dtype=torch.long, device=points.device)
    old = 0
    order = (torch.arange(n, device=points.device) % threads) * n + torch.arange(n, device=points.device) // threads
    for j in range(1, count):
        delta = points - points[old]
        dist = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2]
        temp = torch.minimum(temp, dist)
        max_dist = temp.max()
        old = int(torch.where(temp == max_dist, order, torch.full_like(order, n * (threads + 1))).argmin().item())
        result[j] = old
    return result


def _ball_reference(points: torch.Tensor, centers: torch.Tensor,
                    radius: float, samples: int) -> torch.Tensor:
    """原 CUDA ball_query：d²==0 或 0<=d²<r²，按原点序截断。

    首次命中值填满整行，再由后续命中覆盖前面的槽位；无命中时保留
    初始化的全零行。索引无梯度。
    """
    output = torch.zeros((centers.shape[0], samples), dtype=torch.long, device=points.device)
    for i, center in enumerate(centers):
        delta = points - center
        dist = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2]
        hit = torch.where((dist == 0) | ((dist >= 0) & (dist < radius * radius)))[0][:samples]
        if hit.numel():
            output[i].fill_(int(hit[0]))
            output[i, :hit.numel()] = hit
    return output


def _impl(indices_3d: torch.Tensor, indices_2d: torch.Tensor,
          features_2d: torch.Tensor, fps_num: int, radius: float,
          max_cluster_samples: int, dist_thresh: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Args: indices [M,5] 为 (batch,mix,z,y,x)；features_2d [M,64]。

    Returns: 仅 2D 的补齐索引 [Q,5]、补齐特征 [Q,64] 和 long [Q]
    参考 LiDAR 全体素行号，未匹配为 -1。原 N4 在进入 fps_NN_fast
    前对缺少 2D 独有体素的 batch 补一个零特征占位；此步骤保留补齐。
    使用各尺度配置的 fps_num/radius/max_cluster_samples/dist_thresh。
    离散索引、FPS、ball query 不可微；返回特征对原 2D 特征可微。
    输出 Q 依输入 mask 动态变化，无准确 Meta。
    """
    only_2d = indices_2d[:, 1] == 0
    query_indices = indices_2d[only_2d]
    query_features = features_2d[only_2d]
    batch_size = int(torch.unique(indices_3d[:, 0]).numel())
    present = torch.unique(query_indices[:, 0])
    for batch_id in range(batch_size):
        if not bool((present == batch_id).any()):
            pad_indices = indices_2d.new_zeros((1, indices_2d.shape[1]))
            pad_indices[0, 0] = batch_id
            query_indices = torch.cat([query_indices, pad_indices], dim=0)
            query_features = torch.cat([query_features, features_2d.new_zeros((1, features_2d.shape[1]))], dim=0)

    nn_indices = torch.full((query_indices.shape[0],), -1, dtype=torch.long, device=query_indices.device)
    lidar_coords = indices_3d[:, [0, 2, 3, 4]]
    query_coords = query_indices[:, [0, 2, 3, 4]]
    base = 0
    for batch_id in range(batch_size):
        qmask = query_coords[:, 0] == batch_id
        kmask = lidar_coords[:, 0] == batch_id
        query = query_coords[qmask][:, 1:].float()
        key = lidar_coords[kmask][:, 1:].float()
        per_query = torch.full((query.shape[0],), -1, dtype=torch.long, device=query.device)
        if query.shape[0] <= fps_num:
            distances = torch.norm(query[:, None, :] - key[None, :, :], p=2, dim=-1)
            values, indices = distances.min(dim=-1)
            valid = values < dist_thresh
            per_query[valid] = indices[valid]
        else:
            centers = query[_fps_reference(query.contiguous(), fps_num)]
            distances = torch.norm(centers[:, None, :] - key[None, :, :], p=2, dim=-1)
            values, indices = distances.min(dim=-1)
            valid = values < dist_thresh
            groups = _ball_reference(query.contiguous(), centers.contiguous(), radius, max_cluster_samples)
            expanded_indices = indices[:, None].repeat(1, max_cluster_samples).reshape(-1)
            expanded_valid = valid[:, None].repeat(1, max_cluster_samples).reshape(-1)
            per_query[groups.reshape(-1)[expanded_valid]] = expanded_indices[expanded_valid]
        nonempty = per_query != -1
        per_query[nonempty] += base
        nn_indices[qmask] = per_query
        # 行号属于整批 LiDAR 特征，须累计此前所有 batch 的行数。
        base += int(kmask.sum())
    return query_indices, query_features, nn_indices



_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("msmdfusion_lidar_reference_assignment", _impl)

def msmdfusion_lidar_reference_assignment(indices_3d: torch.Tensor, indices_2d: torch.Tensor,
                               features_2d: torch.Tensor, fps_num: int, radius: float,
                               max_cluster_samples: int, dist_thresh: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """为相机独有体素分配同批 LiDAR 参考行。

    Args: 索引均为 int32 [M,5] (batch,mix,z,y,x) 体素坐标；
        features_2d 为 [M2,64]；后四个标量是 FPS 数、球半径、最多
        邻居数和最近邻距离阈值，距离单位为体素格。
    Returns: 补齐后的相机索引 [Q,5]、特征 [Q,64]、全批 LiDAR 行号
        long [Q]；未匹配为 -1。输出与输入同 device。Q 数据依赖，
        PyTorch 2.0 不注册不准确的 Meta；离散索引不可微。
    """
    return torch.ops.fusionrepo.msmdfusion_lidar_reference_assignment(
        indices_3d, indices_2d, features_2d, fps_num, radius, max_cluster_samples, dist_thresh)
