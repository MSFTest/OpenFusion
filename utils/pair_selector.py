import torch
from torch.library import Library




def pair_selector(iou_matrix: torch.Tensor,
                  iou_threshold: float = 0.0) -> torch.Tensor:
    """
    根据 2D IoU 筛选有效的 (2D_i, 3D_j) 配对，并返回其索引。

    Args:
        iou_matrix: (K, N)
            2D IoU 矩阵，第 i 行对应第 i 个 2D 框，第 j 列对应第 j 个 3D 投影框。
        iou_threshold: float
            IoU 选择阈值，只有 IoU > 阈值 的 (i, j) 才被选为有效配对。

    Returns:
        pair_indices: (p, 2)
            每行表示一个有效配对，格式为 (i_2d, j_3d)，共 p 个配对。
    """

    K, N = iou_matrix.shape

    # 找出所有满足 IoU > threshold 的位置
    valid = (iou_matrix > iou_threshold)

    # nonzero() 得到所有 (i,j) 坐标
    idx = valid.nonzero(as_tuple=False)  # [p, 2]

    return idx


# ============================================================
# 注册实现
# ============================================================
_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("pair_selector", pair_selector)


# ============================================================
# Meta 实现（等价 register_fake）
# ============================================================
def _pair_selector_meta(iou_matrix: torch.Tensor,
                        iou_threshold: float = 0.0):
    # meta 模式下无法计算 p 的真实大小，返回空形状 [0,2]
    return torch.empty((0, 2), device="meta", dtype=torch.long)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("pair_selector", _pair_selector_meta)