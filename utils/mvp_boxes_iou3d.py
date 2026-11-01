import torch
from torch.library import Library
from utils import operator_registry as _registry
import numpy as np
from utils._mvp_extension import load_extension
def to_pcdet(boxes):
    boxes = boxes[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes[:, -1] = -boxes[:, -1] - np.pi / 2
    return boxes
def _impl(boxes_a, boxes_b):
    '原CenterPoint/PCDet CUDA三维IoU，保留原多边形边界与epsilon。\nArgs:\n    boxes_a/boxes_b: 同一CUDA设备float32[N,7]/[M,7]，原LiDAR中心xyz,w,l,h,yaw。\n    距离米、yaw弧度；先交换w/l并转换yaw=-yaw-pi/2后传原PCDet核。\nReturns:\n    float32[N,M]，同设备；BEV核不可微，高度和体积保留原张量梯度。本项目仅no_grad目标路径调用；支持空N/M。非contiguous输入在核前转连续。\n    只支持float32；非法dtype/device在进入会终止进程的原C++核前明确拒绝。\n'
    if boxes_a.dtype!=torch.float32 or boxes_b.dtype!=torch.float32:raise TypeError('MVP IoU requires float32')
    if boxes_a.device!=boxes_b.device:raise ValueError('MVP IoU requires a shared device')
    if boxes_a.ndim!=2 or boxes_b.ndim!=2 or boxes_a.shape[1]!=7 or boxes_b.shape[1]!=7:raise ValueError('MVP IoU requires [N,7]/[M,7]')
    if boxes_a.numel()==0 or boxes_b.numel()==0:
        return boxes_a.new_zeros((len(boxes_a),len(boxes_b)))
    if not boxes_a.is_cuda:raise ValueError('MVP nonempty IoU requires original CUDA extension')
    iou3d_nms_cuda=load_extension()
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    boxes_a = to_pcdet(boxes_a)
    boxes_b = to_pcdet(boxes_b)
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)
    overlaps_bev = boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)
    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)
    overlaps_3d = overlaps_bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-06)
    return iou3d

_lib_impl=Library('fusionrepo','IMPL','CompositeExplicitAutograd')
_lib_impl.impl('mvp_boxes_iou3d',_impl)

def _meta(boxes_a,boxes_b):
    if boxes_a.dtype!=torch.float32 or boxes_b.dtype!=torch.float32:raise TypeError('MVP IoU requires float32')
    return boxes_a.new_empty((boxes_a.shape[0],boxes_b.shape[0]))
_lib_meta=Library('fusionrepo','IMPL','Meta')
_lib_meta.impl('mvp_boxes_iou3d',_meta)

def mvp_boxes_iou3d(boxes_a, boxes_b):
    """调用 fusionrepo::mvp_boxes_iou3d；shape/dtype/device/边界及坐标见 _impl。"""
    return torch.ops.fusionrepo.mvp_boxes_iou3d(boxes_a,boxes_b)
