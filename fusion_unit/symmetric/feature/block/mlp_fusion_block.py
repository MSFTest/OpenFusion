# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MLPFusionBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)