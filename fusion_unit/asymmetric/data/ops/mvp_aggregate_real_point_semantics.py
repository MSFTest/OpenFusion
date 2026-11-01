import torch
from torch.library import Library
from utils import operator_registry as _registry
from utils._mvp_inference_dispatch import register_inference_autograd
from common.indexed_max_reduce import indexed_max_reduce
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


@torch.no_grad()
def _impl(raw_points, point_labels, foreground, num_camera=6):
    """原 virtual_gen.py:163–168，跨相机语义逐通道聚合并标注真实点。

    Args:
        raw_points: float32 [N,4] LiDAR xyzr，XYZ 单位米，原点序。
        point_labels: float32 [C*N,11]，相机主序十类 one-hot 加分数。
        foreground: bool [N]，跨视图前景联合；num_camera: C，原默认 6。
        输入同设备；保留原 cat 的 dtype 提升，不做静默设备迁移。
    Returns:
        输入设备 real_points [F,15]（xyzr 加语义），int64 原始点索引 [F]，不可微。
    每个通道单独取最大值，可得到 multi-hot；不选择单个最佳视图。
    重复坐标点不去重，输出维持原点序。F=0 可为空；N=0 沿用原 reshape 异常。
    F 依赖 foreground 数值，PyTorch 2.0 无准确 Meta，不注册上界形状。
    """
    per_camera = point_labels.reshape(num_camera, raw_points.shape[0], -1)
    labels = indexed_max_reduce(per_camera, 0, False)[0]
    selected = foreground.bool()
    return feature_concatenation([raw_points[selected], labels[selected]], 1), selected.nonzero(as_tuple=False).reshape(-1)


_lib_impl = Library('fusionrepo', 'IMPL', 'CompositeExplicitAutograd')
_lib_impl.impl('mvp_aggregate_real_point_semantics', _impl)
register_inference_autograd('mvp_aggregate_real_point_semantics')


def mvp_aggregate_real_point_semantics(raw_points, point_labels, foreground, num_camera=6):
    """LiDAR xyzr [N,4]、相机主序语义 [CN,11]、前景 [N]→[F,15] 和原索引。

    米制坐标、逐通道最大值、dtype 提升、动态 F 和无梯度契约见 _impl。
    """
    return torch.ops.fusionrepo.mvp_aggregate_real_point_semantics(raw_points, point_labels, foreground, num_camera)
