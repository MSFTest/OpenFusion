"""MSMDFusion 虚拟点权重和硬体素化。

原源码：``MSMDFusionDetector.get_foreground2D``、``voxelize`` 与
``HardSimpleVFE``。正式路径调用当前 mmcv 的确定性 ``Voxelization``；
离散坐标分配不求导，体素特征通过原输入重新聚合，因此 score 与图像特征
保留梯度。坐标为 LiDAR 米制，输出索引布局为 ``(batch,z,y,x)``。
"""

from __future__ import annotations

import torch
from torch import nn

from fusion_unit.symmetric.feature.ops.feature_gate_multiply import feature_gate_multiply

try:
    from mmcv.ops import Voxelization
except ImportError:
    Voxelization = None


def _hard_voxelize_reference(points: torch.Tensor,
                             voxel_size: tuple[float, float, float],
                             max_voxels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按原输入顺序、每 voxel 前 10 点执行可微均值聚合。"""
    lower = (-54.0, -54.0, -5.0)
    upper = (54.0, 54.0, 3.0)
    grid = tuple(round((upper[i] - lower[i]) / voxel_size[i]) for i in range(3))
    groups: dict[tuple[int, int, int], list[int]] = {}
    with torch.no_grad():
        for row, point in enumerate(points):
            coord = tuple(int(torch.floor((point[j] - lower[j]) / voxel_size[j]).item())
                          for j in range(3))
            if any(coord[j] < 0 or coord[j] >= grid[j] for j in range(3)):
                continue
            zyx = (coord[2], coord[1], coord[0])
            if zyx not in groups:
                if len(groups) >= max_voxels:
                    break
                groups[zyx] = []
            if len(groups[zyx]) < 10:
                groups[zyx].append(row)
    if not groups:
        return (points.new_empty((0, points.shape[1])),
                torch.empty((0, 3), dtype=torch.int32, device=points.device))
    features = torch.stack([
        points[torch.tensor(rows, device=points.device)].mean(0)
        for rows in groups.values()
    ])
    indices = torch.tensor(list(groups), dtype=torch.int32, device=points.device)
    return features, indices


def _aggregate_for_native_coordinates(points: torch.Tensor, coordinates: torch.Tensor,
                                      num_points: torch.Tensor,
                                      voxel_size: tuple[float, float, float]) -> torch.Tensor:
    """按 mmcv 选定的 voxel/点数重新聚合，保持原值并恢复 autograd。"""
    if coordinates.numel() == 0:
        return points.new_empty((0, points.shape[1]))
    with torch.no_grad():
        lower = points.new_tensor([-54.0, -54.0, -5.0])
        upper = points.new_tensor([54.0, 54.0, 3.0])
        size = points.new_tensor(voxel_size)
        xyz = torch.floor((points[:, :3] - lower) / size).long()
        grid = torch.round((upper - lower) / size).long()
        valid = ((xyz >= 0) & (xyz < grid)).all(1)
        point_rows = torch.where(valid)[0]
        point_zyx = xyz[valid][:, [2, 1, 0]]
        mul_yx = int(grid[1] * grid[0])
        mul_x = int(grid[0])
        point_keys = point_zyx[:, 0] * mul_yx + point_zyx[:, 1] * mul_x + point_zyx[:, 2]
        voxel_keys = (coordinates[:, 0].long() * mul_yx
                      + coordinates[:, 1].long() * mul_x
                      + coordinates[:, 2].long())
        sorted_keys, order = torch.sort(voxel_keys)
        positions = torch.searchsorted(sorted_keys, point_keys)
        in_range = positions < sorted_keys.numel()
        matched = torch.zeros_like(in_range)
        matched[in_range] = sorted_keys[positions[in_range]] == point_keys[in_range]
        point_rows = point_rows[matched]
        voxel_rows = order[positions[matched]]

        sequence = torch.arange(voxel_rows.numel(), device=voxel_rows.device)
        composite = voxel_rows * (voxel_rows.numel() + 1) + sequence
        grouped = torch.argsort(composite)
        grouped_rows = voxel_rows[grouped]
        group_start_mask = torch.ones_like(grouped_rows, dtype=torch.bool)
        group_start_mask[1:] = grouped_rows[1:] != grouped_rows[:-1]
        starts = torch.where(group_start_mask)[0]
        repeated_starts = torch.repeat_interleave(starts, torch.diff(
            torch.cat([starts, starts.new_tensor([grouped_rows.numel()])])) )
        ranks = torch.arange(grouped_rows.numel(), device=grouped_rows.device) - repeated_starts
        keep = ranks < num_points[grouped_rows].long()
        selected_points = point_rows[grouped][keep]
        selected_voxels = grouped_rows[keep]

    output = points.new_zeros((coordinates.shape[0], points.shape[1]))
    output.index_add_(0, selected_voxels, points[selected_points])
    return output / num_points.to(points.dtype).clamp_min(1).unsqueeze(1)


class VirtualPointVoxelEncoder(nn.Module):
    """共享 score network，并在 1/2/4/8 倍 voxel 上编码相机虚拟点。

    输入 sampled ``[P,49]``、score_inputs ``[P,66]``、foreground_points
    ``[P,15]``、sample_offsets ``[B+1]``。输出 features ``[M,64]`` 和
    int32 indices ``[M,4]``。空样本沿用原实现的 100 个零点填充语义。
    """

    def __init__(self, *, smoke_reference: bool = False):
        super().__init__()
        if not smoke_reference and Voxelization is None:
            raise RuntimeError("MSMDFusion 正式硬体素化需要 mmcv.ops.Voxelization")
        self.smoke_reference = bool(smoke_reference)
        self.backend = "reference" if smoke_reference else "mmcv"
        self.score_net = nn.Sequential(nn.Linear(66, 1), nn.ReLU())

    def _voxelize(self, points: torch.Tensor, downscale: int):
        voxel_size = tuple(v * downscale for v in (0.075, 0.075, 0.2))
        max_voxels = 120000 if self.training else 160000
        if self.smoke_reference:
            return _hard_voxelize_reference(points, voxel_size, max_voxels)
        layer = Voxelization(
            list(voxel_size), [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            10, (120000, 160000), deterministic=True).to(points.device)
        layer.train(self.training)
        with torch.no_grad():
            _, coordinates, counts = layer(points.detach())
        features = _aggregate_for_native_coordinates(
            points, coordinates, counts, voxel_size)
        return features, coordinates.int()

    def forward(self, sampled: torch.Tensor, score_inputs: torch.Tensor,
                foreground_points: torch.Tensor, sample_offsets: torch.Tensor,
                downscale: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = sample_offsets.numel() - 1
        if foreground_points.shape[0] != sampled.shape[0] or sampled.shape[0] != score_inputs.shape[0]:
            raise ValueError("foreground、sampled 与 score_inputs 行数必须一致")
        scores = self.score_net(score_inputs)
        gated = feature_gate_multiply(sampled, scores, False)
        all_features, all_indices = [], []
        for batch_id in range(batch_size):
            start = int(sample_offsets[batch_id].item())
            end = int(sample_offsets[batch_id + 1].item())
            points = torch.cat([foreground_points[start:end], gated[start:end]], dim=1)
            if points.shape[0] == 0:
                points = points.new_zeros((100, foreground_points.shape[1] + sampled.shape[1]))
            features, coordinates = self._voxelize(points, downscale)
            if features.shape[0]:
                scale = features.new_tensor([13.5, 13.5, 2.0])
                features = torch.cat([features[:, :3] / scale, features[:, 3:]], dim=1)
            batch_column = torch.full((coordinates.shape[0], 1), batch_id,
                                      dtype=torch.int32, device=coordinates.device)
            all_features.append(features)
            all_indices.append(torch.cat([batch_column, coordinates], dim=1))
        return torch.cat(all_features), torch.cat(all_indices)
