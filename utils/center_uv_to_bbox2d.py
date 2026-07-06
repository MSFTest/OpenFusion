import torch
from typing import Tuple
from torch.library import Library





def center_uv_to_bbox2d(uv: torch.Tensor,
                        w: torch.Tensor,
                        h: torch.Tensor) -> torch.Tensor:
    """
    将投影得到的 2D 中心点 (u, v) 按给定 w/h 扩展成 2D bbox。

    注意：这里保持与 CLOCs 原始实现一致：
        x1 = u - w
        y1 = v - h
        x2 = u + w
        y2 = v + h

    Args:
        uv: (N, 2)
            2D 中心点像素坐标 (u, v)。
        w: (N,)
            对应 3D 框的宽度（原算子直接拿 boxes_3d_lidar[:, 3]）。
        h: (N,)
            对应 3D 框的高度（原算子直接拿 boxes_3d_lidar[:, 5]）。

    Returns:
        boxes2d: (N, 4)
            2D bbox，格式为 [x1, y1, x2, y2]。
    """
    if uv.dim() != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be (N,2)")
    if w.dim() != 1 or h.dim() != 1:
        raise ValueError("w and h must be (N,)")

    u = uv[:, 0]
    v = uv[:, 1]
    x1 = u - w
    y1 = v - h
    x2 = u + w
    y2 = v + h
    return torch.stack([x1, y1, x2, y2], dim=1).clone()


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("center_uv_to_bbox2d", center_uv_to_bbox2d)


# ============================================================
# meta 实现（等价于 register_fake）
# ============================================================
def _center_uv_to_bbox2d_meta(uv, w, h):
    return torch.empty((uv.shape[0], 4), device="meta", dtype=uv.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("center_uv_to_bbox2d", _center_uv_to_bbox2d_meta)