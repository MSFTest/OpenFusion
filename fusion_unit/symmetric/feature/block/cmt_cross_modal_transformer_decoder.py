# -*- coding: utf-8 -*-
"""CMT 六层 cross-modal Transformer decoder。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from common.transformer_ffn_block import transformer_ffn_block
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class _SelfAttention(nn.Module):
    """原 PETR MultiheadAttention self-attention 包装及外层 residual dropout。"""

    def __init__(self, embed_dims: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, query, query_pos, attn_mask):
        identity = query
        positioned = query + query_pos
        output = self.attn(positioned, positioned, query, attn_mask=attn_mask)[0]
        return identity + self.residual_dropout(output)


class _FlashCrossAttention(nn.Module):
    """原 FlashMHA 正式 kernel，或显式启用的 debug 数学 reference。"""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        residual_dropout: float,
        debug_reference_attention: bool,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads:
            raise ValueError("embed_dims must be divisible by num_heads")
        head_dim = embed_dims // num_heads
        if head_dim % 8 or head_dim > 128:
            raise ValueError("CMT FlashAttention requires head_dim divisible by 8 and <=128")
        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.debug_reference_attention = bool(debug_reference_attention)
        self._flash_kernel = None
        if not self.debug_reference_attention:
            try:
                from flash_attn.flash_attn_interface import (
                    flash_attn_unpadded_kvpacked_func,
                )
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "CMT formal decoder requires the original flash_attn extension; "
                    "install the CMT-compatible flash-attn build, or set "
                    "debug_reference_attention=True only for explicitly named CPU/debug tests"
                ) from error
            if not torch.cuda.is_available():
                raise RuntimeError("CMT formal decoder requires a CUDA device for flash_attn")
            self._flash_kernel = flash_attn_unpadded_kvpacked_func
        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * embed_dims, embed_dims)
        )
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dims))
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.residual_dropout = nn.Dropout(residual_dropout)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, memory, query_pos, memory_pos):
        identity = query
        q = (query + query_pos).transpose(0, 1)
        k = (memory + memory_pos).transpose(0, 1)
        v = memory.transpose(0, 1)
        weights = self.in_proj_weight.chunk(3)
        biases = self.in_proj_bias.chunk(3)
        q, k, v = [F.linear(item, weight, bias) for item, weight, bias in zip((q, k, v), weights, biases)]
        batch, query_len = q.shape[:2]
        key_len = k.shape[1]
        q = q.view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.debug_reference_attention:
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0
            )
        else:
            if not q.is_cuda:
                raise RuntimeError(
                    "CMT formal flash attention requires CUDA query, key and value"
                )
            # 原 FlashMHA 的 projection 和 residual 均为 fp32；仅 inner
            # FlashAttention 通过 auto_fp16 转半精度并以 out_fp32=True 返回。
            q, k, v = q.half(), k.half(), v.half()
            # 原 FlashAttention 在 key_padding_mask=None 时采用 unpadded packed-KV 路径。
            q_unpadded = q.transpose(1, 2).reshape(batch * query_len, self.num_heads, self.head_dim)
            packed_kv = torch.stack((k, v), dim=2).transpose(1, 3).reshape(
                batch * key_len, 2, self.num_heads, self.head_dim
            )
            cu_query = torch.arange(
                0,
                (batch + 1) * query_len,
                step=query_len,
                dtype=torch.int32,
                device=q.device,
            )
            cu_key = torch.arange(
                0,
                (batch + 1) * key_len,
                step=key_len,
                dtype=torch.int32,
                device=q.device,
            )
            context = self._flash_kernel(
                q_unpadded,
                packed_kv,
                cu_query,
                cu_key,
                query_len,
                key_len,
                0.0,
                softmax_scale=None,
                causal=False,
            ).reshape(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
            context = context.float()
        context = context.transpose(1, 2).reshape(batch, query_len, self.embed_dims)
        output = self.out_proj(context).transpose(0, 1)
        return identity + self.residual_dropout(output)


class _FFN(nn.Module):
    """保留原 MMCV FFN 参数层级，并复用公共 FFN+residual+LayerNorm 原语。"""

    def __init__(self, embed_dims: int, feedforward_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(0.0),
        )

    def forward(self, query: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        # 公共原语采用 [B,N,C]；CMT 内部为 [N,B,C]，只做无损布局变换。
        transposed = query.transpose(0, 1)
        output = transformer_ffn_block(
            transposed,
            self.layers[0].weight,
            self.layers[0].bias,
            self.layers[3].weight,
            self.layers[3].bias,
            norm.weight,
            norm.bias,
            norm.eps,
        )
        return output.transpose(0, 1)


class _DecoderLayer(nn.Module):
    """精确执行 self_attn,norm,cross_attn,norm,ffn,norm。"""

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        attention_dropout,
        debug_reference_attention,
    ):
        super().__init__()
        self.self_attn = _SelfAttention(embed_dims, num_heads, attention_dropout)
        self.cross_attn = _FlashCrossAttention(
            embed_dims, num_heads, attention_dropout, debug_reference_attention
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])
        self.ffn = _FFN(embed_dims, feedforward_channels)

    def forward(self, query, memory, query_pos, memory_pos, self_attn_mask):
        query = self.norms[0](self.self_attn(query, query_pos, self_attn_mask))
        query = self.norms[1](self.cross_attn(query, memory, query_pos, memory_pos))
        return self.ffn(query, self.norms[2])


class CrossModalTransformerDecoder(nn.Module):
    """CMT 非共享多层 cross-modal decoder。

    原源码位置：``cmt_transformer.py:CmtTransformer``、``petr_transformer.py``
    与 ``attention.py:FlashMHA``。正式结构为 6 层、C=256、8 heads、FFN=1024；
    每层顺序和参数均独立，唯一 post_norm 在每层输出处共享。BEV memory 始终在前，
    image memory 按 view-major 在后；content query 为零，仅 position query 引导。
    原 FlashMHA 构造把第三位置参数传给 bias，故 inner attention dropout 实际为 0，
    外层 residual dropout 保持配置值。输出 ``[L,B,N,C]``，最后 nan_to_num。

    Args:
        embed_dims: token 通道数，正式配置 256。
        num_heads: self/cross attention 头数，正式配置 8；每头维度须为 8 的倍数
            且不超过 128。
        num_layers: 非共享 decoder 层数，正式配置 6。
        feedforward_channels: FFN 隐层通道，正式配置 1024。
        attention_dropout: self-attention、外层 residual dropout，正式配置 0.1；
            原 cross-attention inner dropout 因源码位置参数语义固定为 0。
        debug_reference_attention: 仅供显式 CPU/debug 测试的 SDPA 数学 reference。
            默认 False，正式模式必须加载原 ``flash_attn`` CUDA 扩展；projection
            保持原 fp32，仅 inner kernel 使用 fp16；缺扩展时明确失败，绝不降级。

    Returns:
        forward 返回 ``[num_layers,B,N,embed_dims]``，dtype/device 跟随正式模型
        或显式 debug 输入；memory 使用 sequence-first contiguous/reshape 布局，
        self-attention mask 为 ``bool[N,N]``，不使用 key padding mask。
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        feedforward_channels: int = 1024,
        attention_dropout: float = 0.1,
        debug_reference_attention: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(
                    self.embed_dims,
                    num_heads,
                    feedforward_channels,
                    attention_dropout,
                    debug_reference_attention,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.post_norm = nn.LayerNorm(self.embed_dims)
        self.debug_reference_attention = bool(debug_reference_attention)
        self.init_weights()

    def init_weights(self) -> None:
        """执行原 CmtTransformer 对所有 dim>1 权重的 Xavier uniform 初始化。"""
        for module in self.modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
                bias = getattr(module, "bias", None)
                if isinstance(bias, torch.Tensor):
                    nn.init.zeros_(bias)

    def prepare_memory(self, bev_feature: torch.Tensor, image_feature: torch.Tensor, views: int):
        """返回 BEV-first 与 view-major image memory，均为 ``[tokens,B,C]``。"""
        batch, channels = bev_feature.shape[:2]
        bev_memory = bev_feature.flatten(2).permute(2, 0, 1)
        image_memory = image_feature.reshape(
            batch, views, channels, image_feature.shape[-2], image_feature.shape[-1]
        ).permute(1, 3, 4, 0, 2).reshape(-1, batch, channels)
        return bev_memory, image_memory

    def forward(
        self,
        bev_feature: torch.Tensor,
        image_feature: torch.Tensor,
        bev_position: torch.Tensor,
        image_position: torch.Tensor,
        query_position: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """融合 BEV/image memory，返回每层 post-norm query。"""
        if bev_feature.ndim != 4 or image_feature.ndim != 4:
            raise ValueError("bev_feature and image_feature must have shape [B,C,H,W]")
        batch, channels, bev_h, bev_w = bev_feature.shape
        views = image_position.shape[1] if image_position.ndim == 5 else 0
        if channels != self.embed_dims or image_feature.shape[:2] != (batch * views, self.embed_dims):
            raise ValueError("feature batch/channel dimensions do not match decoder")
        if query_position.shape[0] != batch or query_position.shape[-1] != self.embed_dims:
            raise ValueError("query_position must have shape [B,N,embed_dims]")
        bev_memory, image_memory = self.prepare_memory(bev_feature, image_feature, views)
        if bev_position.shape != (bev_h * bev_w, self.embed_dims):
            raise ValueError("bev_position must have shape [H*W,embed_dims]")
        bev_memory_position = bev_position.unsqueeze(1).repeat(1, batch, 1)
        expected = (batch, views, image_feature.shape[2], image_feature.shape[3], self.embed_dims)
        if image_position.shape != expected:
            raise ValueError("image_position shape must match [B,V,H,W,embed_dims]")
        image_memory_position = image_position.permute(1, 2, 3, 0, 4).reshape(
            -1, batch, self.embed_dims
        )
        memory = feature_concatenation([bev_memory, image_memory], 0)
        memory_position = feature_concatenation(
            [bev_memory_position, image_memory_position], 0
        )
        query_pos = query_position.transpose(0, 1)
        query = torch.zeros_like(query_pos)
        intermediate = []
        for layer in self.layers:
            query = layer(query, memory, query_pos, memory_position, self_attn_mask)
            intermediate.append(self.post_norm(query))
        return torch.nan_to_num(torch.stack(intermediate).transpose(1, 2))

    @staticmethod
    def original_state_dict_key_map() -> dict:
        """描述原 decoder 层级到当前层级的无损映射规则。"""
        return {
            "decoder.layers.i.attentions.0.attn.*": "layers.i.self_attn.attn.*",
            "decoder.layers.i.attentions.1.attn.in_proj_*": "layers.i.cross_attn.in_proj_*",
            "decoder.layers.i.attentions.1.attn.out_proj.*": "layers.i.cross_attn.out_proj.*",
            "decoder.layers.i.norms.j.*": "layers.i.norms.j.*",
            "decoder.layers.i.ffns.0.layers.*": "layers.i.ffn.layers.*",
            "decoder.post_norm.*": "post_norm.*",
        }

    @staticmethod
    def remap_original_state_dict(original: dict) -> dict:
        """将原 CmtTransformer decoder 权重无损映射到当前 state_dict。"""
        mapped = {}
        for original_key, value in original.items():
            position = original_key.find("decoder.")
            if position < 0:
                continue
            key = original_key[position + len("decoder."):]
            if key.startswith("post_norm."):
                mapped[key] = value
                continue
            if not key.startswith("layers."):
                continue
            parts = key.split(".")
            layer_index = parts[1]
            suffix = ".".join(parts[2:])
            if suffix.startswith("attentions.0.attn."):
                suffix = "self_attn.attn." + suffix[len("attentions.0.attn."):]
            elif suffix.startswith("attentions.1.attn."):
                suffix = "cross_attn." + suffix[len("attentions.1.attn."):]
            elif suffix.startswith("norms."):
                pass
            elif suffix.startswith("ffns.0.layers.0.0."):
                suffix = "ffn.layers.0." + suffix[len("ffns.0.layers.0.0."):]
            elif suffix.startswith("ffns.0.layers.1."):
                suffix = "ffn.layers.3." + suffix[len("ffns.0.layers.1."):]
            else:
                continue
            mapped["layers." + layer_index + "." + suffix] = value
        return mapped
