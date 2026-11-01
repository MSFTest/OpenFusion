import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd


@torch.no_grad()
def _impl(masks, classes, scores, height=900, width=1600):
    """构造 MVP 实例语义，原 virtual_gen.py:178–193、203–228。

    Args:
        masks: 六视图 bool [Mc,H,W] 列表，原图像素布局 H×W。
        classes: 同设备 int64 [Mc] 列表，类别编号 0–9。
        scores: 同设备 float32 [Mc] 列表，实例得分；不额外按分数筛选。
        height,width: 原图高宽，正式默认 900、1600；缩小值仅供显式 smoke。
    Returns:
        float32 [M,W*H+1] 掩码（x 主序，末列零基 camera_id），
        float32 [M,11] 语义（十类 one-hot 加得分），均在输入设备，不可微。
        M 是删除零面积掩码后的实例数，保持相机及实例原顺序，可为零。
    边界：非法类别沿用 gather 异常，不裁剪；六视图可全部无实例。
    Meta/Fake：M 依赖掩码值，PyTorch 2.0 无准确静态 Meta，不注册伪形状。
    """
    device = scores[0].device
    one_hot_labels = []
    for i in range(10):
        label = torch.zeros(10, device=device, dtype=torch.float32)
        label[i] = 1
        one_hot_labels.append(label)
    one_hot_labels = torch.stack(one_hot_labels, dim=0)
    all_masks, all_labels = [], []
    for camera_id in range(len(masks)):
        pred_mask = masks[camera_id].reshape(scores[camera_id].shape[0], width * height)
        empty = pred_mask.sum(dim=1) == 0
        label, score = classes[camera_id][~empty], scores[camera_id][~empty]
        pred_mask = pred_mask[~empty].reshape(-1, height, width).permute(0, 2, 1).reshape(-1, width * height)
        ids = torch.tensor(camera_id, dtype=torch.float32, device=device).reshape(1, 1).repeat(len(label), 1)
        all_masks.append(torch.cat([pred_mask, ids], dim=1))
        semantics = one_hot_labels.gather(0, label.reshape(-1, 1).repeat(1, 10))
        all_labels.append(torch.cat([semantics, score.unsqueeze(-1)], dim=1))
    return torch.cat(all_masks, dim=0), torch.cat(all_labels, dim=0)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_build_instance_semantics', _impl)
register_inference_autograd('mvp_build_instance_semantics')


def mvp_build_instance_semantics(masks, classes, scores, height=900, width=1600):
    """六视图 bool 掩码/int64 类别/float32 分数→x 主序掩码与十一维语义。

    动态 M、输入设备、原实例顺序、空掩码过滤及无梯度契约见 _impl。
    """
    return torch.ops.fusionrepo.mvp_build_instance_semantics(masks, classes, scores, height, width)
