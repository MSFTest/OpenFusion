"""Faraway-Frustum 论文口径评测工具。"""

__all__ = ["evaluate_faraway_kitti", "filter_faraway_anno"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from model.faraway_frustum.evaluation import faraway_kitti_eval

    return getattr(faraway_kitti_eval, name)
