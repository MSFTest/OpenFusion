# -*- coding: utf-8 -*-
"""Shared linear block for VirConv voxel feature mapping."""

from __future__ import annotations

import torch
import torch.nn as nn


class VoxelFeatureLinearBlock(nn.Module):
    """Apply ``Linear -> BatchNorm1d -> ReLU`` to voxel-level features."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(int(in_channels), int(out_channels), bias=False)
        self.bn = nn.BatchNorm1d(int(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, voxel_features: torch.Tensor) -> torch.Tensor:
        x = self.linear(voxel_features)
        x = self.bn(x)
        return self.relu(x)
