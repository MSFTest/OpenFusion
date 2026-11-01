# -*- coding: utf-8 -*-
"""
PointPainting 系统构建入口。
"""

from .build_pointpainting_system import (
    MMDetImageExtractor,
    PointPaintingCriterion,
    PointPaintingPostProcessor,
    PointPaintingSystem,
    build_pointpainting_system,
    test_step,
    train_step,
)

__all__ = [
    "MMDetImageExtractor",
    "PointPaintingCriterion",
    "PointPaintingPostProcessor",
    "PointPaintingSystem",
    "build_pointpainting_system",
    "train_step",
    "test_step",
]
