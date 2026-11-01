"""MVP 原离线 no_grad 的调度辅助；不是算子，不定义 schema。

原源码 virtual_gen.py:73、196；PyTorch 2.0 的 CompositeExplicitAutograd
仍需显式 Autograd 调度，避免给本来不可微的输出附加 NotImplemented 梯度节点。
"""
import torch
from torch.library import Library


_autograd = Library('fusionrepo', 'IMPL', 'Autograd')


def register_inference_autograd(name):
    """为已定义的 fusionrepo 专属算子添加无梯度 redispatch，保留 Meta/Fake 路径。"""
    op = getattr(torch.ops.fusionrepo, name)

    def dispatch(*args, **kwargs):
        with torch.no_grad():
            guard = torch._C._AutoDispatchBelowAutograd()
            try:
                return op(*args, **kwargs)
            finally:
                del guard

    _autograd.impl(name, dispatch)
