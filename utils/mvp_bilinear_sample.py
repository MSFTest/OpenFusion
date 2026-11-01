import torch
from torch.library import Library
from utils import operator_registry as _registry


def _impl(im, x, y):
    '原CenterPoint center_utils.bilinear_interpolate_torch双线性采样。\nArgs:\n    im: 浮点[H,W,C]图，H/W>0，布局[y,x,c]；x/y: 浮点[N]像素坐标，同设备。\n    输入可非contiguous；坐标为连续BEV索引，无半cell偏移，非米。\nReturns:\n    [N,C]，device同im，dtype为im/x/y提升结果。支持空N与对图/坐标的梯度。\n    先clamp整数邻点再计算原权重；边缘/越界可得到零，不等同border grid_sample。\n'
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t(torch.t(Ia) * wa) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

_lib_impl=Library('fusionrepo','IMPL','CompositeExplicitAutograd')
_lib_impl.impl('mvp_bilinear_sample',_impl)
_lib_autograd=Library('fusionrepo','IMPL','Autograd')
_lib_autograd.impl('mvp_bilinear_sample',_impl)

def _meta(im,x,y):
    shape=torch.broadcast_shapes(x.shape,y.shape)
    dtype=torch.promote_types(im.dtype,torch.promote_types(x.dtype,y.dtype))
    return im.new_empty((*shape,im.shape[-1]),dtype=dtype)
_lib_meta=Library('fusionrepo','IMPL','Meta')
_lib_meta.impl('mvp_bilinear_sample',_meta)

def mvp_bilinear_sample(im, x, y):
    """调用 fusionrepo::mvp_bilinear_sample；shape/dtype/device/边界及坐标见 _impl。"""
    return torch.ops.fusionrepo.mvp_bilinear_sample(im,x,y)
