"""融合预测原位回退及分数聚合；源 transfusion_head.py:1007–1029。
Args: torch.ops 的 final/initial 为六字段 Tensor[] 各[B,D,P]；visible bool[B,P] 是实际处理标志；
heatmap[B,10,N] 为初始化热图，indices int64[B,P]，dense[B,10,H,W] 图像引导logits。
Returns: 原位更新且保持别名的 final Tensor[]、scores[B,10,P]、原 dense 别名；
同dtype/device、非连续和空P允许；索引重复保留，越界报错，叶子可微输入原位更新按torch报错。
复用纯 masked_prediction_fallback 后写回原对象；梯度分别到被选融合/初始分支，scores可微。
Python wrapper 收六字段dict并返回源 [dict]；auxiliary仅串最终层，不新增初始auxloss。
Meta/Fake支持固定shape和返回别名，未读取mask值。混合字段dtype不做隐式转换。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry
from fusion_unit.asymmetric.decision.ops.masked_prediction_fallback import masked_prediction_fallback


def _impl(final, initial, visible, heatmap, indices, dense):
    for value, first in zip(final, initial):
        value.copy_(masked_prediction_fallback(value, first, visible))
    scores = heatmap.gather(-1, indices[:, None, :].expand(-1, heatmap.shape[1], -1))
    return final, scores, dense


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transfusion_aggregate_fusion_predictions", _impl)


def _meta(final, initial, visible, heatmap, indices, dense):
    return final, heatmap.new_empty((heatmap.shape[0], heatmap.shape[1], indices.shape[-1])), dense


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transfusion_aggregate_fusion_predictions", _meta)


def transfusion_aggregate_fusion_predictions(final, initial, visible, heatmap, indices, dense, auxiliary=True):
    """聚合六字段dict，输入各[B,D,P]及bool[B,P]；返回源[dict]并原位更新final。
    scores[B,10,P]保持heatmap dtype/device，dense保持输入别名；六字段顺序不变。
    auxiliary=True只拼接最终融合层；可微、边界、坐标约定见本文件模块说明。
    """
    fields = ('center', 'height', 'dim', 'rot', 'vel', 'heatmap')
    values, scores, dense = torch.ops.fusionrepo.transfusion_aggregate_fusion_predictions(
        [final[k] for k in fields], [initial[k] for k in fields], visible, heatmap, indices, dense)
    for key, value in zip(fields, values):
        final[key] = value
    final['query_heatmap_score'] = scores
    final['dense_heatmap'] = dense
    if auxiliary is False:
        return [final]
    return [{k: v if k in ('dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score')
             else torch.cat([v], -1) for k, v in final.items()}]
