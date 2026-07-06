# -*- coding: utf-8 -*-
"""Shared point-level feature channel alignment module."""

from __future__ import annotations

import torch
import torch.nn as nn

from .shared_mlp_1d import SharedMLP1D


class PointFeatureFCAlign(nn.Module):
    """Map point-level features from ``(B, C_in, N)`` to ``(B, C_out, N)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        self.net = SharedMLP1D(
            in_channels=int(in_channels),
            channels=[int(out_channels)],
            use_bn=bool(use_bn),
            activation=bool(activation),
            activate_last=True,
            input_layout="BCL",
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.net(point_features)
