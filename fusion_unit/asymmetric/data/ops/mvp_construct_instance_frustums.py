import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd
from utils.feature_gather_by_pixel_index import feature_gather_by_pixel_index
from common.indexed_max_reduce import indexed_max_reduce


@torch.no_grad()
def _impl(masks, labels, projected, height=900, width=1600):
    """原 virtual_gen.py:66–88、108–110，图像实例选择并赋义 LiDAR 点。

    Args:
        masks: float32 [M,W*H+1]，x 主序，末列零基 camera_id。
        labels: float32 [M,11]，十类 one-hot 加实例得分。
        projected: float32 [C,N,5]，相机主序 u,v,camera_z,valid,camera_id。
        height,width: 原图高宽像素，正式默认 900、1600。
        三个 Tensor 须同设备；坐标为原图像素，Z 单位米。
    Returns:
        valid [CN,M]，dtype 为 masks/projected 的提升类型；int64 首实例 [CN]；
        labels dtype 的逐点语义 [CN,11]；bool 跨相机前景 [N]，均不可微。
    边界：重叠实例取首命中；无命中行索引为零但语义清零；不裁剪越界像素。
    先像素查询再乘相机及可见性，重复像素不合并。M=0 保留原 argmax 异常。
    """
    cameras = projected.shape[0]
    integer_points = projected.reshape(-1, 5)[:, [0, 1, 4]].long()
    chw = masks[:, :-1].reshape(-1, width, height).permute(0, 2, 1)
    valid = feature_gather_by_pixel_index(integer_points[:, :2], chw, 'long', 'CHW', False)
    valid = valid * (masks[:, -1][None] == integer_points[:, -1:])
    valid = valid * projected.reshape(-1, 5)[:, 3:4]
    first = indexed_max_reduce(valid.float(), 1, False)[1]
    point_labels = labels.gather(0, first.reshape(-1, 1).repeat(1, 11))
    point_labels *= valid.sum(dim=1, keepdim=True) > 0
    foreground = (valid.sum(dim=1, keepdim=True) > 0).reshape(cameras, -1).sum(dim=0).bool()
    return valid, first, point_labels, foreground


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_construct_instance_frustums', _impl)


def _meta(masks, labels, projected, height=900, width=1600):
    """固定 CN×M 关联矩阵及逐点输出；不依赖前景非零数量。"""
    if masks.shape[0] == 0:
        raise RuntimeError('原 argmax 对空实例维度报错')
    c, n = projected.shape[:2]
    return (torch.empty((c*n, masks.shape[0]), dtype=torch.promote_types(masks.dtype, projected.dtype), device='meta'),
            torch.empty((c*n,), dtype=torch.long, device='meta'),
            torch.empty((c*n, 11), dtype=labels.dtype, device='meta'),
            torch.empty((n,), dtype=torch.bool, device='meta'))


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('mvp_construct_instance_frustums', _meta)
register_inference_autograd('mvp_construct_instance_frustums')


def mvp_construct_instance_frustums(masks, labels, projected, height=900, width=1600):
    """同设备实例掩码/语义及投影→关联、首实例、逐点语义、前景掩码。

    shape、首命中、越界、相机顺序及不可微契约见 _impl。
    """
    return torch.ops.fusionrepo.mvp_construct_instance_frustums(masks, labels, projected, height, width)
