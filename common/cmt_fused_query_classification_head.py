# -*- coding: utf-8 -*-
"""CMT 六层融合 query 分类预测 head。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from common.cmt_grouped_prediction_layers import (
    GroupedPredictionBranch,
    grouped_to_stacked,
    stacked_queries_to_grouped,
)


class FusedQueryClassificationHead(nn.Module):
    """论文公式(8) ``Psi_cls`` 的原始 grouped SeparateTaskHead 分支。

    原源码位置：``cmt_head.py:SeparateTaskHead`` 与 ``forward_single``。输入为
    ``[num_layers,B,N,hidden_dim]``；每层拥有独立两层 Conv1d 参数，输出字典中
    ``cls_logits[num_layers,B,N-pad,num_classes]``。DN 存在时前 ``pad_size`` 个
    query 原样拆为 ``dn_cls_logits``。末层分类 bias 精确初始化为 -2.19。
    """

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        head_channels: int = 64,
        num_layers: int = 6,
        init_bias: float = -2.19,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.cls_logits = GroupedPredictionBranch(
            self.num_classes, hidden_dim, head_channels, self.num_layers
        )
        nn.init.constant_(self.cls_logits[-1].bias, init_bias)

    def forward(
        self, fused_queries: torch.Tensor, dn_metadata: Optional[Dict[str, object]] = None
    ) -> Dict[str, torch.Tensor]:
        """预测分类 logits，并按原 DN 前缀布局拆分输出。"""
        grouped = stacked_queries_to_grouped(fused_queries, self.num_layers)
        logits = grouped_to_stacked(
            self.cls_logits(grouped), self.num_layers, self.num_classes
        )
        result: Dict[str, torch.Tensor] = {}
        pad_size = int(dn_metadata["pad_size"]) if dn_metadata else 0
        if pad_size > 0:
            result["dn_cls_logits"] = logits[:, :, :pad_size, :]
            logits = logits[:, :, pad_size:, :]
        result["cls_logits"] = logits
        return result

    @staticmethod
    def original_state_dict_key_map() -> dict:
        return {"task_heads.0.cls_logits.*": "cls_logits.*"}

    @staticmethod
    def remap_original_state_dict(original: dict) -> dict:
        """提取任意前缀下的原 task_heads.0.cls_logits 权重。"""
        marker = "task_heads.0.cls_logits."
        return {
            "cls_logits." + key.split(marker, 1)[1]: value
            for key, value in original.items()
            if marker in key
        }
