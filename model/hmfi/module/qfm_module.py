# -*- coding: utf-8 -*-
"""
QFMModule - Query Fusion Module

使用17个算子精确复现HMFI论文的QFM流程。

流程：
    1. QKV Projection (nn.Linear，可学习)
    2. tokens_to_1x1_2d_map → channel_reduce_1x1_2d → qkv_2d_map_to_tokens
    3. attention_tokens_to_batch → multihead_attention_fusion
    4. feature_concatenation + concat_linear_fusion
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QFMModule(nn.Module):
    """
    Query Fusion Module (QFM)

    以LiDAR体素特征为Query，图像体素特征为Key/Value，
    通过多头注意力进行跨模态融合。

    论文对齐：
        - QKV投影: 3个独立Linear层
        - 注意力: 标准Multi-Head Attention
        - 融合: concat([lidar, attn]) → Linear → ReLU

    Args:
        lidar_channels: LiDAR特征通道数
        image_channels: 图像体素特征通道数
        d_k: Query/Key维度
        d_v: Value维度
        num_heads: 注意力头数
        fusion_out_channels: 融合输出通道数
    """

    def __init__(
        self,
        lidar_channels: int,
        image_channels: int,
        d_k: int = 64,
        d_v: int = 64,
        num_heads: int = 4,
        fusion_out_channels: int = 128,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.lidar_channels = lidar_channels
        self.image_channels = image_channels
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.fusion_out_channels = fusion_out_channels
        self.use_residual = bool(use_residual) and lidar_channels == fusion_out_channels

        if d_k % num_heads != 0 or d_v % num_heads != 0:
            raise ValueError("d_k and d_v must be divisible by num_heads")

        self.lidar_norm = nn.LayerNorm(lidar_channels)
        self.image_norm = nn.LayerNorm(image_channels)
        self.attn_norm = nn.LayerNorm(fusion_out_channels)
        self.dropout = nn.Dropout(dropout)

        # ----------------------------------------
        # QKV Projection (可学习参数)
        # ----------------------------------------
        # Query: LiDAR → d_k
        self.q_proj = nn.Linear(lidar_channels, d_k, bias=False)
        # Key/Value: Image → d_k / d_v
        self.k_proj = nn.Linear(image_channels, d_k, bias=False)
        self.v_proj = nn.Linear(image_channels, d_v, bias=False)

        # ----------------------------------------
        # 1x1 Conv投影权重 (用于channel_reduce_1x1_2d)
        # ----------------------------------------
        # 权重形状: (C_out, C_in) - 2D矩阵
        self.register_buffer(
            "q_weight",
            torch.eye(d_k, d_k)  # (d_k, d_k)
        )
        self.register_buffer(
            "k_weight",
            torch.eye(d_k, d_k)  # (d_k, d_k)
        )
        self.register_buffer(
            "v_weight",
            torch.eye(d_v, d_v)  # (d_v, d_v)
        )

        # ----------------------------------------
        # Multi-Head Attention 输出投影
        # ----------------------------------------
        self.out_proj = nn.Linear(d_v, fusion_out_channels, bias=False)

        # ----------------------------------------
        # 融合层: concat + Linear + ReLU
        # ----------------------------------------
        from fusion_unit.symmetric.feature.block.concat_linear_fusion import (
            ConcatLinearFusion
        )

        self.fusion = ConcatLinearFusion(
            in_channels=lidar_channels + fusion_out_channels,
            out_channels=fusion_out_channels,
            concat_dim=-1,
            bias=False,
            use_layernorm=False,
            activation="relu"
        )

    def forward(
        self,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            lidar_features: (N, C_lidar) LiDAR体素特征
            image_features: (M, C_img) 图像体素特征（下采样后）

        Returns:
            fused_features: (N, fusion_out_channels) 融合特征
        """
        if lidar_features.numel() == 0:
            return lidar_features.new_zeros((0, self.fusion_out_channels))

        if image_features.numel() == 0:
            attn_out = lidar_features.new_zeros((lidar_features.shape[0], self.fusion_out_channels))
            return self.fusion([lidar_features, attn_out])

        if image_features.dim() != 2 or lidar_features.dim() != 2:
            raise ValueError("QFM expects lidar_features and image_features to be 2D token tensors")

        # ----------------------------------------
        # Step 1: QKV投影 (N, C) → (N, d_k/d_v)
        # ----------------------------------------
        lidar_in = self.lidar_norm(lidar_features)
        image_in = self.image_norm(image_features)

        query = self.q_proj(lidar_in)   # (N, d_k)
        key = self.k_proj(image_in)     # (M, d_k)
        value = self.v_proj(image_in)   # (M, d_v)

        # ----------------------------------------
        # Step 2: 转1x1 map → 通道压缩 → 转回token
        # ----------------------------------------
        # tokens_to_1x1_2d_map: (N, C) → (1, C, N, 1)
        query_map = torch.ops.fusionrepo.tokens_to_1x1_2d_map(query)
        key_map = torch.ops.fusionrepo.tokens_to_1x1_2d_map(key)
        value_map = torch.ops.fusionrepo.tokens_to_1x1_2d_map(value)

        # channel_reduce_1x1_2d: 1x1卷积
        # 这里使用单位矩阵权重，相当于直接透传
        # 注：如果需要学习额外的通道映射，可在此处添加可学习权重
        query_reduced = torch.ops.fusionrepo.channel_reduce_1x1_2d(
            query_map, self.q_weight
        )
        key_reduced = torch.ops.fusionrepo.channel_reduce_1x1_2d(
            key_map, self.k_weight
        )
        value_reduced = torch.ops.fusionrepo.channel_reduce_1x1_2d(
            value_map, self.v_weight
        )

        # qkv_2d_map_to_tokens: (1, C, N, 1) → (N, C)
        query_t, key_t, value_t = torch.ops.fusionrepo.qkv_2d_map_to_tokens(
            query_reduced, key_reduced, value_reduced
        )

        # ----------------------------------------
        # Step 3: 添加batch维度，准备多头注意力
        # ----------------------------------------
        query_b = torch.ops.fusionrepo.attention_tokens_to_batch(query_t)  # (1, N, d_k)
        key_b = torch.ops.fusionrepo.attention_tokens_to_batch(key_t)      # (1, M, d_k)
        value_b = torch.ops.fusionrepo.attention_tokens_to_batch(value_t)  # (1, M, d_v)

        # ----------------------------------------
        # Step 4: 多头注意力融合
        # ----------------------------------------
        attn_out = torch.ops.fusionrepo.multihead_attention_fusion(
            query_b, key_b, value_b, self.num_heads
        )  # (1, N, d_v)

        # 输出投影
        attn_out = attn_out.squeeze(0)  # (N, d_v)
        attn_out = self.out_proj(attn_out)  # (N, fusion_out_channels)
        attn_out = self.dropout(self.attn_norm(attn_out))

        # ----------------------------------------
        # Step 5: 融合: concat + Linear + ReLU
        # ----------------------------------------
        fused_features = self.fusion([lidar_features, attn_out])
        if self.use_residual:
            fused_features = fused_features + lidar_features

        return fused_features
