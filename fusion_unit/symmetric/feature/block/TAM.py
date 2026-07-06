import torch
import torch.nn as nn


class TranslationAlignmentLayer(nn.Module):
    """
    Translation Alignment Module（TAM，平移对齐算子）。

    该模块用于在 ROI 级别对 LiDAR 特征与伪点云特征进行跨模态对齐，
    通过对两种模态 ROI 特征的联合建模，学习其在空间分布和特征表达
    上的平移偏差，从而缓解跨模态特征不一致问题。

    本模块通过可学习的线性映射与门控机制，对两路特征进行对称的
    校正调整，使其在进入后续注意力与融合算子之前具有更一致的表示。

    本模块属于 **可学习算子**，其参数在训练过程中根据监督信号更新。

    ---
    ### 结构说明（按顺序）
    1. **特征拼接（Concat）**
       - 将 LiDAR ROI 特征与 Pseudo ROI 特征在通道维度进行拼接，
         构建跨模态联合特征表示。
    2. **Linear 映射（2C → C）**
       - 对拼接后的联合特征进行线性映射，学习跨模态的平移校正量。
    3. **Sigmoid 门控**
       - 对校正量进行归一化约束，防止对齐偏移过大。
    4. **对称特征校正**
       - 对 LiDAR 特征进行正向校正，对 Pseudo 特征进行反向校正，
         实现 ROI 级的跨模态对齐。

    ---
    ### 输入
    Args:
        F_lidar_roi (torch.Tensor):
            LiDAR ROI 特征张量，shape 为 **(B, N, C)**
            - B：batch size
            - N：ROI 数量
            - C：ROI 特征维度
        F_pseudo_roi (torch.Tensor):
            Pseudo ROI 特征张量，shape 为 **(B, N, C)**

    ---
    ### 输出
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - F_lidar_aligned：对齐后的 LiDAR ROI 特征，shape 为 **(B, N, C)**
            - F_pseudo_aligned：对齐后的 Pseudo ROI 特征，shape 为 **(B, N, C)**

    ---
    ### 模块用途
    - 在 ROI 级别缓解 LiDAR 与伪点云特征之间的空间与分布不一致问题
    - 为后续 ROI 注意力算子（RAM）提供对齐后的跨模态特征输入
    - 属于 RA-Fusion 阶段中的 **平移对齐核心算子**
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.align_fc = nn.Linear(feat_dim * 2, feat_dim)
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
            F_lidar_aligned (torch.Tensor): (B, N, C)
            F_pseudo_aligned (torch.Tensor): (B, N, C)
        """

        # 跨模态特征拼接
        fused_feat = torch.cat(
            [F_lidar_roi, F_pseudo_roi], dim=-1
        )  # (B, N, 2C)

        # 学习平移对齐偏移量
        offset = self.align_fc(fused_feat)  # (B, N, C)
        offset = self.gate(offset)

        # 对称对齐校正
        F_lidar_aligned = F_lidar_roi + offset
        F_pseudo_aligned = F_pseudo_roi - offset

        return F_lidar_aligned, F_pseudo_aligned
