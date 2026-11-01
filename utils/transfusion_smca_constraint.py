"""SMCA 加性 log Gaussian 几何约束；源 transfusion_head.py:988–1001。
Args: center_xy[P,2]/corner_xy[P,8,2] 为图像像素坐标，visible bool[P]，
grid[1,HW,2] 为 (x+0.5,y+0.5) 特征格坐标，stride 正式4像素/格；同 device。
Returns: query_pos[1,M,2] 与 log_mask[M,HW]，M=visible.sum()，允许 M=0；
位置 dtype 来自 center，mask 跟随浮点距离计算（整数半径的 sigma 默认 float32）。
非连续输入允许；坐标 .int() 截断，半径 ceil，sigma=(2r+1)/6，网格减0.5，
float32 eps 以下置0，log(0)=-inf，不裁剪越界坐标或做深度检查。
位置选择可微、整数中心/半径离散不可微，上游实际几何已 detach。
mask不支持对requires_grad的grid反向：源exp后原位阈值写入也会报版本错误；真实grid无梯度。
PyTorch2.0 无法精确表示数据依赖 M，因此不注册 Meta；Meta/Fake 明确失败，需 eager。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry


def _impl(center_xy, corner_xy, visible, grid, stride=4):
    center_ys = center_xy[visible,1:2]/stride
    center_xs = center_xy[visible,0:1]/stride
    centers = torch.cat([center_xs,center_ys],dim=-1).int()
    span = (corner_xy[visible].max(1).values-corner_xy[visible].min(1).values)/stride
    radius = torch.ceil(span.norm(dim=-1,p=2)/2).int()
    sigma = (radius*2+1)/6.0
    distance = (centers[:,None,:]-(grid-0.5)).norm(dim=-1)**2
    gaussian = (-distance/(2*sigma[:,None]**2)).exp()
    gaussian[gaussian<torch.finfo(torch.float32).eps] = 0
    return torch.cat([center_xs,center_ys],dim=-1)[None],gaussian.log()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transfusion_smca_constraint", _impl)


def transfusion_smca_constraint(center_xy, corner_xy, visible, grid, stride=4):
    """SMCA 加性 log Gaussian 几何约束；源 transfusion_head.py:988–1001。
Args: center_xy[P,2]/corner_xy[P,8,2] 为图像像素坐标，visible bool[P]，
grid[1,HW,2] 为 (x+0.5,y+0.5) 特征格坐标，stride 正式4像素/格；同 device。
Returns: query_pos[1,M,2] 与 log_mask[M,HW]，M=visible.sum()，允许 M=0；
位置 dtype 来自 center，mask 跟随浮点距离计算（整数半径的 sigma 默认 float32）。
非连续输入允许；坐标 .int() 截断，半径 ceil，sigma=(2r+1)/6，网格减0.5，
float32 eps 以下置0，log(0)=-inf，不裁剪越界坐标或做深度检查。
位置选择可微、整数中心/半径离散不可微，上游实际几何已 detach。
mask不支持对requires_grad的grid反向：源exp后原位阈值写入也会报版本错误；真实grid无梯度。
PyTorch2.0 无法精确表示数据依赖 M，因此不注册 Meta；Meta/Fake 明确失败，需 eager。
"""
    return torch.ops.fusionrepo.transfusion_smca_constraint(center_xy, corner_xy, visible, grid, stride)
