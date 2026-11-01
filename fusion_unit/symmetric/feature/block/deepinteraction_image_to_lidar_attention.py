# -*- coding: utf-8 -*-
"""DeepInteraction 图像到 LiDAR 注意力融合块。"""

import torch
from torch import nn


class ImageToLiDARAttention(nn.Module):
    """按有效 token 数分桶执行单头注意力并散射回 BEV。

    Args:
        channels: query/key/value 通道，正式配置 128。
        dropout: MultiheadAttention dropout，正式配置 0.1。
        groups: 有效图像 token 上界，正式为 ``(20,40,80,120)``。
    Returns:
        query ``[M,1,C]``、token ``[M,T*V,C]``、valid ``[M,T*V]``、
        coordinates ``[M,4]=(batch,z,y,x)`` 和 output_shape ``(B,C,H,W)``
        生成 dense ``[B,C,H,W]``。空可见 pillar 为零；重复坐标采用原高级
        索引的后写语义；越界坐标按原 PyTorch 索引行为报错。

    原源码：``encoder_utils.py:216-320`` ``MMRI_I2P``。
    """
    def __init__(self, channels=128, dropout=.1, groups=(20, 40, 80, 120)):
        super().__init__()
        self.groups = tuple(int(value) for value in groups)
        self.align = nn.MultiheadAttention(
            channels, 1, dropout=dropout, kdim=channels, vdim=channels,
            batch_first=True,
        )

    @staticmethod
    def remap_original_state_dict(state_dict):
        """映射原 ``MMRI_I2P.learnedAlign`` checkpoint 键。"""
        return {
            key.replace("learnedAlign.", "align.", 1): value
            for key, value in state_dict.items()
        }

    def _group_attention(self, query, token, valid):
        output = torch.zeros_like(query)
        token_count = valid.sum(-1)
        lower = 0
        for upper in self.groups:
            member = (token_count > lower) & (token_count <= upper)
            if member.any():
                member_token, member_valid = token[member], valid[member]
                compact = member_token.new_zeros(
                    (member_token.shape[0], upper, member_token.shape[-1])
                )
                padding = torch.ones((member_token.shape[0], upper),
                                     dtype=torch.bool, device=member_token.device)
                for row in range(member_token.shape[0]):
                    selected = member_token[row, member_valid[row]]
                    compact[row, :selected.shape[0]] = selected
                    padding[row, :selected.shape[0]] = False
                output[member] = self.align(
                    query[member], compact, compact, key_padding_mask=padding
                )[0]
            lower = upper
        return output

    def forward(self, query, token, valid, coordinates, output_shape):
        aligned = self._group_attention(query, token, valid)
        output = query.new_zeros(output_shape)
        occupied = valid.any(-1)
        coordinate = coordinates[occupied]
        output[coordinate[:, 0].long(), :, coordinate[:, 2].long(),
               coordinate[:, 3].long()] = aligned[occupied, 0]
        return output
