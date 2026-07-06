# -*- coding: utf-8 -*-
import torch
from torch.library import Library




def iou2d_matrix(boxes_a: torch.Tensor,
                 boxes_b: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """
    计算两组 2D 边界框之间的 IoU（Intersection over Union），并输出两两 IoU 矩阵。

    这是跨模型通用的“2D IoU 原语算子”：
        - 只负责 IoU 的数学计算（交并比）
        - 不做 NMS、不做阈值筛选、不做排序、不做 topk
        - 可直接复用在 CLOCs 的 2D-3D 匹配、AVOD 的 NMS 前置 IoU 计算等场景

    IoU 定义：
        inter = area( A ∩ B )
        union = area(A) + area(B) - inter
        iou = inter / union

    Args:
        boxes_a: (K, 4)
            第一组 2D 框，格式为 [x1, y1, x2, y2]。
            其中 (x1, y1) 为左上角，(x2, y2) 为右下角。

        boxes_b: (N, 4)
            第二组 2D 框，格式为 [x1, y1, x2, y2]。

        eps:
            防止除 0 的最小值，默认 1e-6。

    Returns:
        iou: (K, N)
            IoU 矩阵。iou[i, j] 表示 boxes_a 第 i 个框与 boxes_b 第 j 个框的 IoU。

    说明（与各项目原算子关系）：
        - CLOCs 的 compute_iou2d 本质就是该功能：可 100% 直接替换其 IoU 计算部分。
        - AVOD 的 rpn_nms 内部需要“框与框重叠程度”来做抑制：该算子提供 IoU，
          但 NMS 还需要排序/抑制/输出索引（那是后续算子/步骤的事）。
    """
    if boxes_a.dim() != 2 or boxes_a.shape[1] != 4:
        raise ValueError(f"boxes_a must be (K,4), got {tuple(boxes_a.shape)}")
    if boxes_b.dim() != 2 or boxes_b.shape[1] != 4:
        raise ValueError(f"boxes_b must be (N,4), got {tuple(boxes_b.shape)}")

    # 保持与输入一致的 dtype/device，避免精度漂移
    a = boxes_a
    b = boxes_b

    # (K,1,4) and (1,N,4) -> broadcast to (K,N,4)
    a_ = a.unsqueeze(1)
    b_ = b.unsqueeze(0)

    # intersection box
    x1 = torch.max(a_[:, :, 0], b_[:, :, 0])
    y1 = torch.max(a_[:, :, 1], b_[:, :, 1])
    x2 = torch.min(a_[:, :, 2], b_[:, :, 2])
    y2 = torch.min(a_[:, :, 3], b_[:, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # areas
    area_a = (a_[:, :, 2] - a_[:, :, 0]) * (a_[:, :, 3] - a_[:, :, 1])
    area_b = (b_[:, :, 2] - b_[:, :, 0]) * (b_[:, :, 3] - b_[:, :, 1])

    union = area_a + area_b - inter
    iou = inter / torch.clamp(union, min=eps)

    # custom_op 输出建议避免 alias
    return iou.clone()


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("iou2d_matrix", iou2d_matrix)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _iou2d_matrix_meta(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6):
    K = boxes_a.shape[0]
    N = boxes_b.shape[0]
    return torch.empty((K, N), device="meta", dtype=boxes_a.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("iou2d_matrix", _iou2d_matrix_meta)