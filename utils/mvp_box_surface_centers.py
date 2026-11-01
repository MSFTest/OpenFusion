import torch
from torch.library import Library
from utils import operator_registry as _registry
import numpy as np
from torch import stack as tstack
def torch_to_np_dtype(ttype):
    type_map = {torch.float16: np.dtype(np.float16), torch.float32: np.dtype(np.float32), torch.float16: np.dtype(np.float64), torch.int32: np.dtype(np.int32), torch.int64: np.dtype(np.int64), torch.uint8: np.dtype(np.uint8)}
    return type_map[ttype]

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([tstack([rot_cos, -rot_sin]), tstack([rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners
def _impl(boxes):
    '原TwoStageDetector.get_box_center(num_point=5)五点几何。\nArgs:\n    boxes: float32[N,9]，原LiDAR中心xyz,w,l,h,vx,vy,yaw；米/弧度，同设备。\nReturns:\n    float32[5N,3]，按中心/前/后/左/右分段；原corner顺序，无padding，支持空N。\n    计算可微；非contiguous输入保持原.view/einsum语义。仅float32为MVP正式契约。\n'
    corners=center_to_corner_box2d(boxes[:,:2],boxes[:,3:5],boxes[:,-1])
    points=[boxes[:,:3]]
    for i,j in ((0,1),(2,3),(0,3),(1,2)):
        points.append(torch.cat([(corners[:,i]+corners[:,j])/2,boxes[:,2:3]],dim=-1))
    return torch.cat(points,dim=0)


_lib_impl=Library('fusionrepo','IMPL','CompositeExplicitAutograd')
_lib_impl.impl('mvp_box_surface_centers',_impl)
_lib_autograd=Library('fusionrepo','IMPL','Autograd')
_lib_autograd.impl('mvp_box_surface_centers',_impl)

def _meta(boxes):
    torch_to_np_dtype(boxes.dtype)
    return boxes.new_empty((5*boxes.shape[0],3))
_lib_meta=Library('fusionrepo','IMPL','Meta')
_lib_meta.impl('mvp_box_surface_centers',_meta)

def mvp_box_surface_centers(boxes):
    """调用 fusionrepo::mvp_box_surface_centers；shape/dtype/device/边界及坐标见 _impl。"""
    return torch.ops.fusionrepo.mvp_box_surface_centers(boxes)
