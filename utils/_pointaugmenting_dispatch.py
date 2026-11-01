"""PointAugmenting 私有 no_grad 调度辅助；不定义算子或 schema。

原 loading.py 为 NumPy 预处理，两个 detector 的图像装饰在 no_grad 中。
Autograd 层向下重新调度，使 Meta/Fake 使用各算子的 Meta 内核。
"""
import torch
from torch.library import Library


_autograd = Library('fusionrepo', 'IMPL', 'Autograd')


def register_no_grad(name):
    """保留原不可微边界，不给输出附加 NotImplemented 反向节点。"""
    op = getattr(torch.ops.fusionrepo, name)

    def dispatch(*args, **kwargs):
        with torch.no_grad():
            guard = torch._C._AutoDispatchBelowAutograd()
            try:
                return op(*args, **kwargs)
            finally:
                del guard

    _autograd.impl(name, dispatch)


def check_numpy_float(*tensors):
    """原几何输入为 NumPy 可表示的实浮点；bfloat16 不得在 Meta 中伪装可用。"""
    for tensor in tensors:
        if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError('PointAugmenting 几何输入需要 float16/float32/float64')
