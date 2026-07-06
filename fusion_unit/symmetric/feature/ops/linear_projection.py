# -*- coding: utf-8 -*-

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.library import Library


def _linear_projection_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
):
    return F.linear(x, weight, bias)


# CPU / CUDA 显式注册
_lib_impl = Library("fusionrepo", "IMPL")

_lib_impl.impl(
    "linear_projection",
    _linear_projection_impl,
    "CPU"
)

_lib_impl.impl(
    "linear_projection",
    _linear_projection_impl,
    "CUDA"
)


# Meta
_lib_meta = Library("fusionrepo", "IMPL", "Meta")

def _linear_projection_meta(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
):
    out_features = weight.shape[0]
    shape = list(x.shape)
    shape[-1] = out_features
    return torch.empty(tuple(shape), device="meta", dtype=x.dtype)

_lib_meta.impl(
    "linear_projection",
    _linear_projection_meta
)


def linear_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor
):
    return torch.ops.fusionrepo.linear_projection(x, weight, bias)