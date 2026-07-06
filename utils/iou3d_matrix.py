import torch
from torch.library import Library




def iou3d_matrix(boxes_a: torch.Tensor,
                 boxes_b: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """
    计算两组 3D 边界框之间的 IoU（Intersection over Union），并输出两两 IoU 矩阵。

    本算子遵循你在 CLOCs 中给出的 3D IoU 计算方式（近似版本）：
        IoU_3D ≈ IoU_BEV × IoU_height

    其中：
        - IoU_BEV：俯视图 (x, y) 平面上的 2D IoU（忽略 yaw，按 axis-aligned 矩形计算）
        - IoU_height：高度方向 z 的 1D IoU（上下重叠比例）

    说明：
        - 该算子仅负责 IoU 计算，不做 NMS/筛选/排序
        - 对于 CLOCs 等“以该近似 IoU3D 为准”的模型，可直接无缝复用
        - 对于需要更精确 yaw IoU 的模型，可在后续扩展为 oriented IoU（不在此通用算子范围内）

    Args:
        boxes_a: (M, 7)
            第一组 3D 框，格式为 [x, y, z, w, l, h, yaw]。
            其中：
                (x, y, z) 为中心点坐标
                w 为宽（x 方向尺寸）
                l 为长（y 方向尺寸）
                h 为高（z 方向尺寸）
                yaw 为朝向角（本算子近似版本不使用 yaw）

        boxes_b: (N, 7)
            第二组 3D 框，格式同上。

        eps:
            防止除 0 的最小值，默认 1e-6。

    Returns:
        iou3d: (M, N)
            3D IoU 矩阵。iou3d[i, j] 表示 boxes_a 第 i 个框与 boxes_b 第 j 个框的 3D IoU。

    与原算子关系（小白版）：
        - CLOCs 原 iou3d 算子本质就是“输出两两 IoU3D”，本通用算子功能相同，可直接替换。
        - MPCF / AVOD 等里的 NMS/筛选流程需要“重叠程度”，本算子可作为其中的重叠度计算模块，
          但完整 NMS 还需额外的排序、阈值抑制与索引输出逻辑。
    """
    if boxes_a.dim() != 2 or boxes_a.shape[1] != 7:
        raise ValueError(f"boxes_a must be (M,7), got {tuple(boxes_a.shape)}")
    if boxes_b.dim() != 2 or boxes_b.shape[1] != 7:
        raise ValueError(f"boxes_b must be (N,7), got {tuple(boxes_b.shape)}")

    a = boxes_a
    b = boxes_b

    # (M,1,7) and (1,N,7) -> broadcast
    a_ = a.unsqueeze(1)
    b_ = b.unsqueeze(0)

    # -----------------------------
    # BEV (axis-aligned) overlap
    # -----------------------------
    ax, ay, az = a_[:, :, 0], a_[:, :, 1], a_[:, :, 2]
    aw, al, ah = a_[:, :, 3], a_[:, :, 4], a_[:, :, 5]

    bx, by, bz = b_[:, :, 0], b_[:, :, 1], b_[:, :, 2]
    bw, bl, bh = b_[:, :, 3], b_[:, :, 4], b_[:, :, 5]

    a_x1 = ax - aw * 0.5
    a_x2 = ax + aw * 0.5
    a_y1 = ay - al * 0.5
    a_y2 = ay + al * 0.5

    b_x1 = bx - bw * 0.5
    b_x2 = bx + bw * 0.5
    b_y1 = by - bl * 0.5
    b_y2 = by + bl * 0.5

    inter_x1 = torch.max(a_x1, b_x1)
    inter_y1 = torch.max(a_y1, b_y1)
    inter_x2 = torch.min(a_x2, b_x2)
    inter_y2 = torch.min(a_y2, b_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_l = (inter_y2 - inter_y1).clamp(min=0)
    inter_bev = inter_w * inter_l

    area_a_bev = (a_x2 - a_x1).clamp(min=0) * (a_y2 - a_y1).clamp(min=0)
    area_b_bev = (b_x2 - b_x1).clamp(min=0) * (b_y2 - b_y1).clamp(min=0)

    union_bev = area_a_bev + area_b_bev - inter_bev
    iou_bev = inter_bev / torch.clamp(union_bev, min=eps)

    # -----------------------------
    # Height overlap (1D IoU)
    # -----------------------------
    a_z1 = az - ah * 0.5
    a_z2 = az + ah * 0.5
    b_z1 = bz - bh * 0.5
    b_z2 = bz + bh * 0.5

    inter_z1 = torch.max(a_z1, b_z1)
    inter_z2 = torch.min(a_z2, b_z2)

    inter_h = (inter_z2 - inter_z1).clamp(min=0)
    len_a = (a_z2 - a_z1).clamp(min=0)
    len_b = (b_z2 - b_z1).clamp(min=0)

    union_h = len_a + len_b - inter_h
    iou_h = inter_h / torch.clamp(union_h, min=eps)

    # -----------------------------
    # Approx 3D IoU
    # -----------------------------
    iou3d = iou_bev * iou_h
    return iou3d.clone()


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("iou3d_matrix", iou3d_matrix)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _iou3d_matrix_meta(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6):
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]
    return torch.empty((M, N), device="meta", dtype=boxes_a.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("iou3d_matrix", _iou3d_matrix_meta)