import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd


@torch.no_grad()
def _impl(masks, num_virtual=50, height=900, width=1600):
    """原 virtual_gen.py:90–106 的逐实例随机像素采样，不重设全局 RNG。

    Args:
        masks: 输入设备 float32 [M,W*H+1]，x 主序实例掩码加 camera_id。
        num_virtual: 每实例采样数，正式默认 50；零采样仍调用原 randperm。
        height,width: 原图高宽像素，默认 900、1600。
    Returns:
        同设备 float32 [M*num_virtual,3]，列为 u,v,一基实例编号；
        float32 [M*num_virtual,1] 相机编号。离散、不可微，无参数或状态。
    边界：nonzero 枚举及 randperm 顺序与原实现相同；不足时重复首个已选像素。
    M=0 或零面积实例的直接调用保留原 stack/索引异常，调用者须先过滤空掩码。
    """
    camera_ids = masks[:, -1]
    offsets = []
    for mask in masks[:, :-1]:
        indices = mask.reshape(width, height).nonzero()
        selected = torch.randperm(len(indices), device=masks.device)[:num_virtual]
        if len(selected) < num_virtual:
            selected = torch.cat([selected, selected[selected.new_zeros(num_virtual - len(selected))]])
        offsets.append(indices[selected])
    offsets = torch.stack(offsets, dim=0)
    ids = torch.arange(1, 1 + masks.shape[0], dtype=torch.float32, device=masks.device)
    ids = ids.reshape(masks.shape[0], 1, 1).repeat(1, num_virtual, 1)
    pixels = torch.cat([offsets, ids], dim=-1).reshape(-1, 3)
    cameras = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1)
    return pixels, cameras


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_sample_instance_pixels', _impl)


def _meta(masks, num_virtual=50, height=900, width=1600):
    """仅承诺上游已过滤零面积实例后的固定采样数，不伪造空实例成功。"""
    if masks.shape[0] == 0:
        raise RuntimeError('原实例列表为空时 torch.stack 报错')
    q = masks.shape[0] * num_virtual
    return (torch.empty((q, 3), dtype=torch.float32, device='meta'),
            torch.empty((q, 1), dtype=masks.dtype, device='meta'))


_lib_meta = Library('fusionrepo', 'IMPL', 'Meta')
_lib_meta.impl('mvp_sample_instance_pixels', _meta)
register_inference_autograd('mvp_sample_instance_pixels')


def mvp_sample_instance_pixels(masks, num_virtual=50, height=900, width=1600):
    """x 主序 float32 [M,W*H+1]→[M*num_virtual,3] 像素及相机列。

    全局 RNG、补点、空掩码异常及无梯度契约见 _impl。
    """
    return torch.ops.fusionrepo.mvp_sample_instance_pixels(masks, num_virtual, height, width)
