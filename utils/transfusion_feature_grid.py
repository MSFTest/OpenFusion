"""TransFusion create_2D_grid（transfusion_head.py:758）的位置准备适配。
Args: anchor 只提供目标 device，不参与梯度；rows/cols 为行/列非负数量。
Returns: [1,rows*cols,2]，(列+0.5,行+0.5)，行优先，无量纲特征格坐标；
默认浮点 dtype，先 CPU 再迁移 device，原 view/permute 非连续布局（空维允许）。
复用 compute_query_position；正式 BEV180x180、图像112x200及列1x1200。
无参数、buffer 或训练状态；坐标不依赖 anchor 值。Meta/Fake 保持 shape/dtype/stride。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils.bevdet_compute_query_position import compute_query_position


def _impl(anchor, rows, cols):
    if rows < 0 or cols < 0:
        raise RuntimeError("原linspace steps必须非负")
    indices = torch.arange(rows * cols, device="cpu")[None]
    grid = compute_query_position(indices, rows, cols).to(torch.get_default_dtype())
    if rows * cols == 0:
        return grid.new_empty((1, 2, 0)).permute(0, 2, 1).to(anchor.device)
    return grid.transpose(1, 2).contiguous().transpose(1, 2).to(anchor.device)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transfusion_feature_grid", _impl)


def _meta(anchor, rows, cols):
    if rows < 0 or cols < 0:
        raise RuntimeError("原linspace steps必须非负")
    return torch.empty((1, 2, rows * cols), dtype=torch.get_default_dtype(), device="meta").permute(0, 2, 1)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transfusion_feature_grid", _meta)


def transfusion_feature_grid(anchor, rows, cols):
    """TransFusion create_2D_grid（transfusion_head.py:758）的位置准备适配。
Args: anchor 只提供目标 device，不参与梯度；rows/cols 为行/列非负数量。
Returns: [1,rows*cols,2]，(列+0.5,行+0.5)，行优先，无量纲特征格坐标；
默认浮点 dtype，先 CPU 再迁移 device，原 view/permute 非连续布局（空维允许）。
复用 compute_query_position；正式 BEV180x180、图像112x200及列1x1200。
无参数、buffer 或训练状态；坐标不依赖 anchor 值。Meta/Fake 保持 shape/dtype/stride。
"""
    return torch.ops.fusionrepo.transfusion_feature_grid(anchor, rows, cols)
