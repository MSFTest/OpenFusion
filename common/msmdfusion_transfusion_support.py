"""TransFusionHead 实际可达的私有固定依赖。来源：原项目
mmdet3d/core/bbox/coders/transfusion_bbox_coder.py TransFusionBBoxCoder 8–126；
mmdet3d/core/bbox/assigners/hungarian_assigner.py HungarianAssigner3D 92–153；
mmdet3d/core/utils/gaussian.py 5–85；mmdet3d/models/utils/clip_sigmoid.py 4；
原依赖 mmcv.cnn.ConvModule、mmdet FocalLoss/GaussianFocalLoss/L1Loss/
FocalLossCost/PseudoSampler。与目标配置无关的 image-fuse 和 NMS 变体
不在本私有文件构建；目标配置 fuse_img=False 且 nms_type=None。
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from utils.msmdfusion_transfusion_geometry import box3d_iou, gaussian_radius, draw_heatmap_gaussian


def build_conv_layer(cfg, *args, **kwargs):
    """原 mmcv.cnn.build_conv_layer：本配置只实例化 Conv1d/Conv2d。"""
    kind = cfg['type']
    if kind == 'Conv1d':
        return nn.Conv1d(*args, **kwargs)
    if kind == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    raise ValueError(kind)


def kaiming_init(layer, a=0, nonlinearity='relu'):
    """原 mmcv kaiming_init 默认正态初始化、bias=0。"""
    nn.init.kaiming_normal_(layer.weight, a=a, nonlinearity=nonlinearity)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


class ConvModule(nn.Module):
    """原 mmcv.cnn.ConvModule 默认 conv/norm/ReLU，保留 conv/bn/activate 键。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias='auto', conv_cfg=None, norm_cfg=None):
        super().__init__()
        if bias == 'auto':
            bias = norm_cfg is None
        self.conv = build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, bias=bias)
        if norm_cfg is None:
            self.bn = None
        elif norm_cfg['type'] == 'BN1d':
            self.bn = nn.BatchNorm1d(out_channels)
        elif norm_cfg['type'] == 'BN2d':
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            raise ValueError(norm_cfg)
        self.activate = nn.ReLU(inplace=True)
        kaiming_init(self.conv)
        if self.bn is not None:
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.activate(x)


def _cast_float32(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, list):
        return [_cast_float32(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_cast_float32(x) for x in value)
    if isinstance(value, dict):
        return {k: _cast_float32(v) for k, v in value.items()}
    return value


def force_fp32(apply_to=()):
    """原 mmcv.runner.force_fp32 的目标参数转换；本头 loss 转 preds_dicts。"""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            import inspect
            names = list(inspect.signature(fn).parameters)
            values = list(args)
            for name in apply_to:
                if name in kwargs:
                    kwargs[name] = _cast_float32(kwargs[name])
                elif name in names and names.index(name) < len(values):
                    values[names.index(name)] = _cast_float32(values[names.index(name)])
            return fn(*values, **kwargs)
        return wrapped
    return decorate


def clip_sigmoid(x, eps=1e-4):
    """原 mmdet3d/models/utils/clip_sigmoid.py 4–16：inplace sigmoid。"""
    return torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)


def _reduce(loss, weight, avg_factor):
    if weight is not None:
        loss = loss * weight
    return loss.sum() / avg_factor if avg_factor is not None else loss.mean()


class _FocalLoss(nn.Module):
    """mmdet sigmoid focal loss，gamma=2、alpha=.25、reduction=mean。"""

    def __init__(self, cfg):
        super().__init__()
        self.gamma = cfg['gamma']
        self.alpha = cfg['alpha']
        self.loss_weight = cfg.get('loss_weight', 1.0)

    def forward(self, logits, labels, weight=None, avg_factor=None):
        classes = logits.shape[1]
        target = F.one_hot(labels.long(), num_classes=classes + 1)[:, :classes].to(logits.dtype)
        prob = logits.sigmoid()
        pt = (1 - prob) * target + prob * (1 - target)
        focal = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none') * focal
        if weight is not None:
            weight = weight.reshape(-1, 1)
        return self.loss_weight * _reduce(loss, weight, avg_factor)


class _GaussianFocalLoss(nn.Module):
    """mmdet GaussianFocalLoss：正点 target=1，负点按 (1-target)^4 加权。"""

    def __init__(self, cfg):
        super().__init__()
        self.loss_weight = cfg.get('loss_weight', 1.0)

    def forward(self, pred, target, weight=None, avg_factor=None):
        eps = 1e-12
        pos = -(pred + eps).log() * (1 - pred).pow(2) * target.eq(1)
        neg = -(1 - pred + eps).log() * pred.pow(2) * (1 - target).pow(4)
        return self.loss_weight * _reduce(pos + neg, weight, avg_factor)


class _L1Loss(nn.Module):
    """mmdet L1Loss，按 code_weights 加权后以正样本数归一。"""

    def __init__(self, cfg):
        super().__init__()
        self.loss_weight = cfg.get('loss_weight', 1.0)

    def forward(self, pred, target, weight=None, avg_factor=None):
        if target.numel() == 0:
            return pred.sum() * 0
        return self.loss_weight * _reduce((pred - target).abs(), weight, avg_factor)


class _VarifocalLoss(nn.Module):
    """原默认 loss_iou 被构建但实际 forward 中未调用；保留状态属性。"""

    def __init__(self, cfg):
        super().__init__()
        self.loss_weight = cfg.get('loss_weight', 1.0)

    def forward(self, pred, target, weight=None, avg_factor=None):
        prob = pred.sigmoid()
        focal_weight = target * (target > 0).float() + 0.75 * (prob - target).abs().pow(2) * (target <= 0).float()
        return self.loss_weight * _reduce(F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight,
                                          weight, avg_factor)


def build_loss(cfg):
    return {'FocalLoss': _FocalLoss, 'GaussianFocalLoss': _GaussianFocalLoss,
            'L1Loss': _L1Loss, 'VarifocalLoss': _VarifocalLoss}[cfg['type']](cfg)


class TransFusionBBoxCoder:
    """原 transfusion_bbox_coder.py TransFusionBBoxCoder 8–126。"""

    def __init__(self, pc_range, out_size_factor, voxel_size,
                 post_center_range=None, score_threshold=None, code_size=8):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, boxes):
        targets = boxes.new_zeros((boxes.shape[0], self.code_size))
        targets[:, 0] = (boxes[:, 0] - self.pc_range[0]) / (self.out_size_factor * self.voxel_size[0])
        targets[:, 1] = (boxes[:, 1] - self.pc_range[1]) / (self.out_size_factor * self.voxel_size[1])
        targets[:, 3:6] = boxes[:, 3:6].log()
        targets[:, 2] = boxes[:, 2] + boxes[:, 5] * .5
        targets[:, 6] = torch.sin(boxes[:, 6])
        targets[:, 7] = torch.cos(boxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = boxes[:, 7:]
        return targets

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        """保留原源码 39–126 行对 center/dim 的 inplace 改写。"""
        scores, labels = heatmap.max(1)
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * .5
        yaw = torch.atan2(rot[:, 0:1, :], rot[:, 1:2, :])
        pieces = [center, height, dim, yaw] + ([] if vel is None else [vel])
        boxes = torch.cat(pieces, dim=1).permute(0, 2, 1)
        results = []
        for i in range(heatmap.shape[0]):
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            if filter:
                if self.post_center_range is None:
                    raise NotImplementedError('post_center_range is required')
                bounds = torch.tensor(self.post_center_range, device=boxes.device)
                mask &= (boxes[i, :, :3] >= bounds[:3]).all(-1)
                mask &= (boxes[i, :, :3] <= bounds[3:]).all(-1)
                # 原源码以 score_threshold 的真值决定是否实际应用阈值。
                if self.score_threshold:
                    mask &= scores[i] > self.score_threshold
            results.append(dict(bboxes=boxes[i, mask], scores=scores[i, mask], labels=labels[i, mask]))
        return results


def build_bbox_coder(cfg):
    return TransFusionBBoxCoder(**{k: v for k, v in cfg.items() if k != 'type'})


@dataclass
class AssignResult:
    """mmdet AssignResult 被本头实际读取的完整字段。"""
    num_gts: int
    gt_inds: torch.Tensor
    max_overlaps: torch.Tensor | None
    labels: torch.Tensor | None = None


@dataclass
class SamplingResult:
    pos_inds: torch.Tensor
    neg_inds: torch.Tensor
    pos_assigned_gt_inds: torch.Tensor
    pos_gt_bboxes: torch.Tensor


class PseudoSampler:
    """原 mmdet PseudoSampler：正样本 gt_inds>0，负样本 gt_inds==0。"""

    def sample(self, assign_result, bboxes, gt_bboxes):
        pos = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
        neg = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)
        gt_index = assign_result.gt_inds[pos] - 1
        return SamplingResult(pos, neg, gt_index, gt_bboxes[gt_index])


def build_sampler(cfg):
    if cfg['type'] != 'PseudoSampler':
        raise ValueError(cfg['type'])
    return PseudoSampler()


class HungarianAssigner3D:
    """原 mmdet3d/core/bbox/assigners/hungarian_assigner.py 92–153。

    0.15 focal cost + 0.25 归一化 BEV L1 + 0.25 旋转 3D IoU cost；
    scipy linear_sum_assignment 在 CPU 求解，结果回到预测设备。
    """

    def __init__(self, cls_cost, reg_cost, iou_cost, iou_calculator):
        self.cls_cost, self.reg_cost, self.iou_cost = cls_cost, reg_cost, iou_cost

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        num_gt, num_pred = gt_bboxes.shape[0], bboxes.shape[0]
        assigned = bboxes.new_full((num_pred,), -1, dtype=torch.long)
        labels = bboxes.new_full((num_pred,), -1, dtype=torch.long)
        if num_gt == 0 or num_pred == 0:
            if num_gt == 0:
                assigned[:] = 0
            return AssignResult(num_gt, assigned, None, labels)
        prob = cls_pred[0].T.sigmoid()
        gamma, alpha = self.cls_cost['gamma'], self.cls_cost['alpha']
        neg = -(1 - prob + 1e-12).log() * (1 - alpha) * prob.pow(gamma)
        pos = -(prob + 1e-12).log() * alpha * (1 - prob).pow(gamma)
        cost_cls = (pos[:, gt_labels] - neg[:, gt_labels]) * self.cls_cost['weight']
        start = bboxes.new_tensor(train_cfg['point_cloud_range'][:2])
        span = bboxes.new_tensor(train_cfg['point_cloud_range'][3:5]) - start
        cost_reg = torch.cdist((bboxes[:, :2] - start) / span,
                               (gt_bboxes[:, :2] - start) / span, p=1) * self.reg_cost['weight']
        iou = box3d_iou(bboxes, gt_bboxes)
        cost = cost_cls + cost_reg - iou * self.iou_cost['weight']
        rows, cols = linear_sum_assignment(cost.detach().cpu())
        rows = torch.from_numpy(rows).to(bboxes.device)
        cols = torch.from_numpy(cols).to(bboxes.device)
        assigned[:] = 0
        assigned[rows] = cols + 1
        labels[rows] = gt_labels[cols]
        overlap = torch.zeros_like(iou.max(1).values)
        overlap[rows] = iou[rows, cols]
        return AssignResult(num_gt, assigned, overlap, labels)


def build_assigner(cfg):
    if cfg['type'] != 'HungarianAssigner3D':
        raise ValueError(cfg['type'])
    return HungarianAssigner3D(cfg['cls_cost'], cfg['reg_cost'], cfg['iou_cost'], cfg['iou_calculator'])


def multi_apply(fn, *args, **kwargs):
    """原 mmdet.core.multi_apply：map 后按返回值位置聚合为 tuple[list]。"""
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    return tuple(map(list, zip(*map(fn, *args))))
