import torch
import torch.nn as nn


class RPNHead(nn.Module):
    """
    RPNHead 模块（AVOD 融合阶段的候选框生成预测算子）。

    该模块对锚框级融合特征进行判别与回归，
    用于预测每个三维锚框是否为前景目标（objectness），
    以及对应的三维框位置与尺寸修正量。

    本模块属于可学习算子，其参数在训练过程中通过反向传播进行更新。

    ---
    ### 结构说明
    1. **Flatten**
       - 将锚框级特征展平成向量
    2. **全连接层（FC）**
       - 学习锚框级语义表示
    3. **前景置信度预测分支**
       - 输出每个锚框的前景得分
    4. **三维框回归分支**
       - 输出锚框的三维位置与尺寸偏移量

    ---
    ### 输入
    Args:
        anchor_features (torch.Tensor):
            锚框级融合特征，
            shape 为 (N, C, H, W)

    ---
    ### 输出
    Returns:
        objectness_scores (torch.Tensor):
            锚框前景置信度，shape 为 (N, 1)
        bbox_deltas (torch.Tensor):
            三维框回归偏移量，shape 为 (N, 7)
            对应 [dx, dy, dz, ddx, ddy, ddz]
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()

        self.flatten = nn.Flatten()

        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.objectness_fc = nn.Linear(hidden_dim, 1)
        self.bbox_reg_fc = nn.Linear(hidden_dim, 7)

    def forward(self, anchor_features: torch.Tensor):
        x = self.flatten(anchor_features)
        x = self.shared_fc(x)

        objectness_scores = self.objectness_fc(x)
        bbox_deltas = self.bbox_reg_fc(x)

        return objectness_scores, bbox_deltas
