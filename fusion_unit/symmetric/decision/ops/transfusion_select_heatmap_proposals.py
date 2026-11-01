"""融合两路热图并保留局部峰值后，按原降序 argsort 选 query（源 forward_single:843–864）。
Args: 两路浮点 logits[B,10,H,W]，同 device，dtype 可提升、允许非连续；
proposals 原 Python 切片数量（默认200），kernel 正式3；H/W 至少 kernel。
Returns: 类别/空间索引 int64[B,K]，断梯度 heatmap[B,10,H*W]，设备同输入；
热图先 sigmoid 再相加/2，边缘为零，8/9 类不抑制，同位置跨类重复允许，
并列排序遵循当前 torch.argsort，不引入稳定排序或去重。kernel1 保留原错误。
Meta/Fake 支持固定 shape；空 batch 支持，零空间/类别不足10不属源有效输入。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry
from fusion_unit.symmetric.feature.ops.feature_fusion_broadcast_weighted_sum import feature_fusion_broadcast_weighted_sum
from utils.bevdet_local_max_heatmap_nms import local_max_heatmap_nms
from utils.descending_argsort_topk import descending_argsort_topk


def _impl(lidar_logits, image_logits, proposals=200, kernel=3):
    # 公共NMS会变为连续布局；先保留源.view不支持转置H/W的错误边界。
    torch.empty_like(lidar_logits, device='meta').view(lidar_logits.shape[0], lidar_logits.shape[1], -1)
    if lidar_logits.shape[1] < 10:
        raise IndexError("原 nuScenes 选择必须可索引类别8/9")
    heatmap = feature_fusion_broadcast_weighted_sum(torch.stack((lidar_logits.detach().sigmoid(), image_logits.detach().sigmoid())), None) / 2
    if kernel // 2 == 0:
        # 源 padding:-padding 为 0:0，此参数原本报错，不沿用公共 NMS 的 kernel1 特例。
        raise RuntimeError("原 TransFusion kernel=1 内部切片为空，无法写入池化结果")
    heatmap = local_max_heatmap_nms(heatmap, kernel, 0).view(heatmap.shape[0], heatmap.shape[1], -1)
    total = heatmap.shape[1] * heatmap.shape[2]
    count = min(proposals, total) if proposals >= 0 else max(total + proposals, 0)
    top, _ = descending_argsort_topk(heatmap, count)
    return top // heatmap.shape[-1], top % heatmap.shape[-1], heatmap


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transfusion_select_heatmap_proposals", _impl)


def _meta(lidar_logits, image_logits, proposals=200, kernel=3):
    torch.empty_like(lidar_logits).view(lidar_logits.shape[0], lidar_logits.shape[1], -1)
    if lidar_logits.shape[1] < 10:
        raise IndexError("原 nuScenes 选择必须可索引类别8/9")
    if min(lidar_logits.shape[-2:]) < kernel:
        raise RuntimeError("原max_pool2d空间尺寸不足")
    if kernel // 2 == 0:
        raise RuntimeError("原 TransFusion kernel=1 内部切片为空")
    b, c, h, w = lidar_logits.shape
    total = c * h * w
    count = min(proposals, total) if proposals >= 0 else max(total + proposals, 0)
    dtype = torch.promote_types(lidar_logits.dtype, image_logits.dtype)
    return (torch.empty((b, count), dtype=torch.long, device="meta"),
            torch.empty((b, count), dtype=torch.long, device="meta"),
            torch.empty((b, c, h*w), dtype=dtype, device="meta"))


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transfusion_select_heatmap_proposals", _meta)


def transfusion_select_heatmap_proposals(lidar_logits, image_logits, proposals=200, kernel=3):
    """融合两路热图并保留局部峰值后，按原降序 argsort 选 query（源 forward_single:843–864）。
Args: 两路浮点 logits[B,10,H,W]，同 device，dtype 可提升、允许非连续；
proposals 原 Python 切片数量（默认200），kernel 正式3；H/W 至少 kernel。
Returns: 类别/空间索引 int64[B,K]，断梯度 heatmap[B,10,H*W]，设备同输入；
热图先 sigmoid 再相加/2，边缘为零，8/9 类不抑制，同位置跨类重复允许，
并列排序遵循当前 torch.argsort，不引入稳定排序或去重。kernel1 保留原错误。
Meta/Fake 支持固定 shape；空 batch 支持，零空间/类别不足10不属源有效输入。
"""
    return torch.ops.fusionrepo.transfusion_select_heatmap_proposals(lidar_logits, image_logits, proposals, kernel)
