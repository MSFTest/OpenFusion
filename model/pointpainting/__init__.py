# -*- coding: utf-8 -*-
"""
PointPainting 重构项目。
"""

from .pointpainting_framework import PointPaintingFramework
from .initialization.build_pointpainting_system import (
    MMDetImageExtractor,
    PointPaintingCriterion,
    PointPaintingPostProcessor,
    PointPaintingSystem,
    build_pointpainting_system,
    test_step,
    train_step,
)

__all__ = [
    "PointPaintingFramework",
    "MMDetImageExtractor",
    "PointPaintingCriterion",
    "PointPaintingPostProcessor",
    "PointPaintingSystem",
    "build_pointpainting_system",
    "train_step",
    "test_step",
]
