# -*- coding: utf-8 -*-
import torch
from torch.library import Library




def nms3d(
    boxes_3d: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """
    对 3D 检测框执行 NMS（Non-Maximum Suppression）。

    功能：
        根据每个 3D 框的分数进行排序，
        使用 IoU3D 抑制高度重叠的框，
        最终返回保留下来的框索引。

    NMS 规则：
        1. 按分数从高到低排序
        2. 依次取当前最高分框
        3. 删除与其 IoU3D > 阈值的框
        4. 重复直到结束

    Args:
        boxes_3d: (N, 7)
            3D 检测框
            [x, y, z, w, l, h, yaw]

        scores: (N,)
            每个 3D 框的置信度分数

        iou_threshold: float
            IoU3D 阈值，大于该值则抑制

    Returns:
        keep_indices: (M,)
            保留下来的框索引
    """

    if boxes_3d.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_3d.device)

    # 按分数排序
    order = torch.argsort(scores, descending=True)

    keep = []

    while order.numel() > 0:

        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        current_box = boxes_3d[i].unsqueeze(0)
        other_boxes = boxes_3d[order[1:]]

        # ⭐ 调用第三个通用算子
        ious = torch.ops.fusionrepo.iou3d_matrix(
            current_box,
            other_boxes
        ).squeeze(0)

        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.stack(keep)


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("nms3d", nms3d)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _nms3d_meta(boxes_3d, scores, iou_threshold):
    return torch.empty((0,), device="meta", dtype=torch.long)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("nms3d", _nms3d_meta)