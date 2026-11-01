import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd
from common.indexed_max_reduce import indexed_max_reduce


@torch.no_grad()
def _impl(virtual_points, virtual_cameras, valid, projected, dist_thresh=3000.):
    """原 virtual_gen.py:108–136 的实例约束最近邻深度匹配。

    Args:
        virtual_points: float32 [S,3]，原图 u,v,一基实例编号；末列被原地乘 10000。
        virtual_cameras: float32 [S,1] 零基相机编号。
        valid: float32 [CN,M] 实例关联；projected: float32 [C,N,5] 相机主序投影。
        dist_thresh: 编码坐标欧氏距离的严格上限，默认 3000；不是米制距离。
        输入同设备；离散选择不可微，无模型参数。
    Returns:
        pixels [Q,3]、cameras [Q,1]、相机 Z 深度（米）[Q]、int64 CN 索引 [Q]。
        无真实前景时四项均 None；存在前景但全被阈值过滤时四个空 Tensor。
    保留原范数运算、并列首索引、输入原地修改、严格阈值与排序。
    Q 和 Optional 分支依赖数据，PyTorch 2.0 不注册伪静态 Meta。
    """
    points_xyc = projected.reshape(-1, 5)[:, [0, 1, 4]]
    valid_mask = valid.sum(dim=1) > 0
    real_ids = (indexed_max_reduce(valid.float(), 1, False)[1] + 1)[valid_mask]
    real_points = torch.cat([points_xyc[:, :2][valid_mask], real_ids[..., None]], dim=-1)
    real_points[:, -1] *= 1e4
    virtual_points[:, -1] *= 1e4
    if len(real_points) == 0:
        return None, None, None, None
    distance = torch.norm(virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1)
    negative_distance, nearest = indexed_max_reduce(-distance, 1, False)
    keep = -negative_distance < dist_thresh
    indices = valid_mask.nonzero(as_tuple=False).reshape(-1)
    nearest = indices[nearest[keep]]
    return virtual_points[keep], virtual_cameras[keep], projected.reshape(-1, 5)[nearest, 2], nearest


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_match_virtual_pixel_depths', _impl)
register_inference_autograd('mvp_match_virtual_pixel_depths')


def mvp_match_virtual_pixel_depths(virtual_points, virtual_cameras, valid, projected, dist_thresh=3000.):
    """同设备虚拟像素及真实投影关联→筛选像素、相机、米制深度、CN 索引。

    注意 virtual_points 末列原地乘 10000；动态 Q/None 及无梯度语义见 _impl。
    """
    return torch.ops.fusionrepo.mvp_match_virtual_pixel_depths(virtual_points, virtual_cameras, valid, projected, dist_thresh)
