# -*- coding: utf-8 -*-
"""CMT 六层融合 query 三维框回归 head。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from common.cmt_grouped_prediction_layers import (
    GroupedPredictionBranch,
    grouped_to_stacked,
    stacked_queries_to_grouped,
)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """原 mmdetection inverse_sigmoid，边界先 clamp 到 ``[0,1]``。"""
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class FusedQueryBoxRegressionHead(nn.Module):
    """论文公式(8) ``Psi_reg`` 的五个原始 grouped 回归分支。

    原源码位置：``cmt_head.py:SeparateTaskHead/forward_single``。输入
    ``[num_layers,B,N,hidden_dim]`` 与归一化 ``reference_points[B,N,3]``；输出
    center(2)、height(1)、dim(3)、rot(2)、vel(2)。center/height 加 reference
    inverse-logit 后 sigmoid 并转换为米制 LiDAR 坐标。保留原实际字段拼接语义，
    不擅自修正 bbox coder 对 height/dim 的解释。pc_range 为普通 tuple，非 buffer。
    """

    OUTPUTS = {"center": 2, "height": 1, "dim": 3, "rot": 2, "vel": 2}

    def __init__(
        self,
        hidden_dim: int = 256,
        head_channels: int = 64,
        num_layers: int = 6,
        pc_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        for name, channels in self.OUTPUTS.items():
            setattr(self, name, GroupedPredictionBranch(channels, hidden_dim, head_channels, num_layers))
        self.pc_range = tuple(float(value) for value in pc_range)

    def forward(
        self,
        fused_queries: torch.Tensor,
        reference_points: torch.Tensor,
        dn_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        """返回五字段 normal/DN 预测，shape 均为 ``[L,B,N,C_field]``。"""
        if reference_points.ndim != 3 or reference_points.shape[-1] != 3:
            raise ValueError("reference_points must have shape [B,N,3]")
        grouped = stacked_queries_to_grouped(fused_queries, self.num_layers)
        outputs: Dict[str, torch.Tensor] = {}
        for name, channels in self.OUTPUTS.items():
            outputs[name] = grouped_to_stacked(
                getattr(self, name)(grouped), self.num_layers, channels
            )
        reference = inverse_sigmoid(reference_points.clone())
        center = (outputs["center"] + reference[None, :, :, :2]).sigmoid()
        height = (outputs["height"] + reference[None, :, :, 2:3]).sigmoid()
        pc = center.new_tensor(self.pc_range)
        metric_center = torch.zeros_like(center)
        metric_height = torch.zeros_like(height)
        metric_center[..., 0:1] = center[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        metric_center[..., 1:2] = center[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        metric_height[..., 0:1] = height[..., 0:1] * (pc[5] - pc[2]) + pc[2]
        outputs["center"], outputs["height"] = metric_center, metric_height
        result: Dict[str, torch.Tensor] = {}
        pad_size = int(dn_metadata["pad_size"]) if dn_metadata else 0
        for name, value in outputs.items():
            if pad_size > 0:
                result["dn_" + name] = value[:, :, :pad_size, :]
                value = value[:, :, pad_size:, :]
            result[name] = value
        return result

    @staticmethod
    def original_state_dict_key_map() -> dict:
        return {"task_heads.0." + name + ".*": name + ".*" for name in FusedQueryBoxRegressionHead.OUTPUTS}

    @classmethod
    def remap_original_state_dict(cls, original: dict) -> dict:
        """提取任意前缀下五个原 task head 分支的权重。"""
        mapped = {}
        for branch in cls.OUTPUTS:
            marker = "task_heads.0." + branch + "."
            for key, value in original.items():
                if marker in key:
                    mapped[branch + "." + key.split(marker, 1)[1]] = value
        return mapped
