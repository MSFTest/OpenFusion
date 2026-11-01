"""以实际处理标志选择更新预测或初始预测，供 TransFusion/DeepInteraction 的字段回退调用。
Args: updated/initial 同 dtype/device 的 [B,D,P] Tensor，允许非连续和空维；
processed 同 device 的 bool[B,P]，True 为真正完成更新（不能用仅几何可见替代）。
Returns: 新的连续 [B,D,P]，dtype/device 保持；不修改、不返回输入别名，不改变 query 顺序。
未选中的 NaN 不污染结果；梯度只到被选分支；mask 不可微。Meta/Fake 无数据依赖 shape。
调用者保留项目规定的原位写回，公共算子不偷偷更改项目状态。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry


def _impl(updated, initial, processed):
    if updated.shape != initial.shape or updated.ndim != 3:
        raise ValueError("预测必须同 shape [B,D,P]")
    if processed.dtype != torch.bool or processed.shape != (updated.shape[0], updated.shape[2]):
        raise ValueError("processed 必须是 bool[B,P]")
    if updated.dtype != initial.dtype or updated.device != initial.device or processed.device != updated.device:
        raise ValueError("两路预测必须同 dtype/device，mask 必须同 device")
    return torch.where(processed[:, None, :], updated, initial)


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("masked_prediction_fallback", _impl)


def _meta(updated, initial, processed):
    return _impl(updated, initial, processed)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("masked_prediction_fallback", _meta)


def masked_prediction_fallback(updated, initial, processed):
    """以实际处理标志选择更新预测或初始预测，供 TransFusion/DeepInteraction 的字段回退调用。
Args: updated/initial 同 dtype/device 的 [B,D,P] Tensor，允许非连续和空维；
processed 同 device 的 bool[B,P]，True 为真正完成更新（不能用仅几何可见替代）。
Returns: 新的连续 [B,D,P]，dtype/device 保持；不修改、不返回输入别名，不改变 query 顺序。
未选中的 NaN 不污染结果；梯度只到被选分支；mask 不可微。Meta/Fake 无数据依赖 shape。
调用者保留项目规定的原位写回，公共算子不偷偷更改项目状态。
"""
    return torch.ops.fusionrepo.masked_prediction_fallback(updated, initial, processed)
