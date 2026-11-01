import torch
from torch.library import Library
from utils import operator_registry as _registry
import numpy as np
from utils._mvp_extension import load_extension

def _impl(boxes, scores, thresh, pre_maxsize=1000, post_max_size=83):
    '原CenterPoint rotate_nms_pcdet：原排序、坐标转换与CUDA rotated NMS。\nArgs:\n    boxes: 同设备CUDA float32[N,7]，原LiDAR中心xyz,w,l,h,yaw，米/弧度。\n    scores: float32[N]；thresh: BEV IoU阈值；pre_maxsize/post_max_size为前后候选上限。\nReturns:\n    int64[K]原输入索引，同设备，按原torch.sort结果排列；相同分数保留原排序语义。\n    K依赖数据，torch2.0不注册虚假Meta。空N返回空索引；离散NMS不可微。\n'
    if boxes.dtype!=torch.float32 or scores.dtype!=torch.float32:raise TypeError('MVP NMS requires float32')
    if boxes.device!=scores.device:raise ValueError('MVP NMS requires a shared device')
    if boxes.ndim!=2 or boxes.shape[1]!=7 or scores.shape!=(len(boxes),):raise ValueError('MVP NMS requires boxes[N,7],scores[N]')
    if not boxes.is_cuda:raise ValueError('MVP NMS requires original CUDA extension')
    iou3d_nms_cuda = load_extension()
    boxes = boxes[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes[:, -1] = -boxes[:, -1] - np.pi / 2
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    if len(boxes) == 0:
        num_out = 0
    else:
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    selected = order[keep[:num_out].to(device=boxes.device)].contiguous()
    if post_max_size is not None:
        selected = selected[:post_max_size]
    return selected

_lib_impl=Library('fusionrepo','IMPL','CompositeExplicitAutograd')
_lib_impl.impl('mvp_rotated_nms',_impl)

def mvp_rotated_nms(boxes, scores, thresh, pre_maxsize=1000, post_max_size=83):
    """调用 fusionrepo::mvp_rotated_nms；shape/dtype/device/边界及坐标见 _impl。"""
    return torch.ops.fusionrepo.mvp_rotated_nms(boxes,scores,thresh,pre_maxsize,post_max_size)
