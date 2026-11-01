# -*- coding: utf-8 -*-
"""Faraway-Frustum 系统构建入口。"""

from model.faraway_frustum.initialization.build_faraway_frustum_system import (
    FarawayFrustumSystem,
    build_faraway_frustum_system,
    test_step,
    train_step,
)

__all__ = [
    "FarawayFrustumSystem",
    "build_faraway_frustum_system",
    "train_step",
    "test_step",
]
