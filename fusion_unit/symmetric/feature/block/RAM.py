import torch
import torch.nn as nn


class ROIAttentionLayer(nn.Module):
    """
    ROI Attention Module（RAM，ROI 注意力算子）。

    该模块用于在 ROI 级别对跨模态特征进行重要性建模，通过可学习的
    注意力机制，为不同 ROI 以及不同模态分配自适应权重，从而突出
    可靠特征并抑制噪声或不可信的 ROI 表达。

    RAM 接收已经完成空间对齐的 LiDAR 与伪点云 ROI 特征作为输入，
    在 ROI 维度上学习其相对重要性，为后续特征融合提供加权后的输入。

    本模块属于 **可学习算子**，其参数在训练过程中根据监督信号更新。

    ---
    ### 结构说明（按顺序）
    1. **特征拼接（Concat）**
       - 将对齐后的 LiDAR ROI 特征与 Pseudo ROI 特征在通道维拼接，
         构建 ROI 级跨模态联合表示。
    2. **Linear 映射（2C → C）**
       - 对联合特征进行线性变换，提取 ROI 级语义表示。
    3. **Sigmoid 注意力生成**
       - 生成取值范围在 [0, 1] 的 ROI 注意力权重。
    4. **特征加权**
       - 使用注意力权重对两路模态特征进行逐 ROI 加权，
         强化重要 ROI，抑制不可靠 ROI。

    ---
    ### 输入
    Args:
        F_lidar_roi (torch.Tensor):
            对齐后的 LiDAR ROI 特征张量，shape 为 **(B, N, C)**
            - B：batch size
            - N：ROI 数量
            - C：ROI 特征维度
        F_pseudo_roi (torch.Tensor):
            对齐后的 Pseudo ROI 特征张量，shape 为 **(B, N, C)**

    ---
    ### 输出
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - F_lidar_weighted：加权后的 LiDAR ROI 特征，shape 为 **(B, N, C)**
            - F_pseudo_weighted：加权后的 Pseudo ROI 特征，shape 为 **(B, N, C)**

    ---
    ### 模块用途
    - 在 ROI 级别建模跨模态特征的重要性权重
    - 抑制噪声 ROI，对高置信度 ROI 进行特征增强
    - 为后续特征拼接与融合算子提供加权后的输入
    - 属于 RA-Fusion 阶段中的 **ROI 注意力核心算子**
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.attention_fc = nn.Linear(feat_dim * 2, feat_dim)
        self.gate = nn.Sigmoid()

    def forward(
        self,
        F_lidar_roi: torch.Tensor,
        F_pseudo_roi: torch.Tensor
    ):
        """
        Args:
            F_lidar_roi (torch.Tensor): (B, N, C)
            F_pseudo_roi (torch.Tensor): (B, N, C)

        Returns:
            F_lidar_weighted (torch.Tensor): (B, N, C)
            F_pseudo_weighted (torch.Tensor): (B, N, C)
        """

        # 跨模态 ROI 特征拼接
        fused_feat = torch.cat(
            [F_lidar_roi, F_pseudo_roi], dim=-1
        )  # (B, N, 2C)

        # 生成 ROI 注意力权重
        attn_weight = self.attention_fc(fused_feat)  # (B, N, C)
        attn_weight = self.gate(attn_weight)

        # ROI 级特征加权
        F_lidar_weighted = F_lidar_roi * attn_weight
        F_pseudo_weighted = F_pseudo_roi * attn_weight

        return F_lidar_weighted, F_pseudo_weighted
