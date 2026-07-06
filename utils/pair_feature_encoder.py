# utils/pair_feature_encoder.py 直接替换成这个版本
import torch
from torch.library import Library


def pair_feature_encoder(
    pair_indices: torch.Tensor,
    iou_matrix: torch.Tensor,
    boxes_3d: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fusionrepo.pair_feature_encoder(
        pair_indices, iou_matrix, boxes_3d, scores_2d, scores_3d
    )


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")


def pair_feature_encoder_impl(
    pair_indices: torch.Tensor,
    iou_matrix: torch.Tensor,
    boxes_3d: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
) -> torch.Tensor:
    # 输出 (P,4): [iou2d, score2d, score3d, distance3d]
    if pair_indices.numel() == 0:
        return torch.empty((0, 4), device=iou_matrix.device, dtype=iou_matrix.dtype)

    i = pair_indices[:, 0].long()
    j = pair_indices[:, 1].long()

    iou = iou_matrix[i, j].to(dtype=iou_matrix.dtype).view(-1, 1)          # (P,1)
    s2  = scores_2d[i].to(dtype=iou_matrix.dtype).view(-1, 1)              # (P,1)
    s3  = scores_3d[j].to(dtype=iou_matrix.dtype).view(-1, 1)              # (P,1)

    # distance3d: 用 3D box center 的欧氏距离（也可以只用 x,z）
    b3d = boxes_3d[j].to(dtype=iou_matrix.dtype)
    if b3d.dim() != 2:
        b3d = b3d.view(b3d.shape[0], -1)
    # 取 center (x,y,z)
    cx, cy, cz = b3d[:, 0], b3d[:, 1], b3d[:, 2]
    dist = torch.sqrt(cx * cx + cy * cy + cz * cz).view(-1, 1)
    dist = dist / 80.0
    dist = torch.clamp(dist, 0.0, 1.0)

    return torch.cat([iou, s2, s3, dist], dim=1)                            # (P,4)


_lib_impl.impl("pair_feature_encoder", pair_feature_encoder_impl)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")


def pair_feature_encoder_meta(
    pair_indices: torch.Tensor,
    iou_matrix: torch.Tensor,
    boxes_3d: torch.Tensor,
    scores_2d: torch.Tensor,
    scores_3d: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((pair_indices.shape[0], 4), device="meta", dtype=iou_matrix.dtype)


_lib_meta.impl("pair_feature_encoder", pair_feature_encoder_meta)