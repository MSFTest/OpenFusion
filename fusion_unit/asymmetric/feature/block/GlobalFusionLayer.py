import torch
import torch.nn as nn


class GlobalFusionLayer(nn.Module):
    """
    GlobalFusionLayer 模块（LoGoNet 全局融合阶段核心算子）。

    该模块在场景级别对点云特征与图像特征进行跨模态融合，用于为 voxel / proposal
    引入来自图像的语义信息。其核心思想是：以点云特征为引导，在图像特征图中
    动态采样与其空间位置对应的图像特征，并完成跨模态特征对齐与融合。

    本模块属于 **可学习算子**，其内部包含线性映射与注意力权重相关参数，
    在训练过程中通过反向传播进行更新。

    ---
    ### 结构说明（按顺序）
    1. **Linear：点云特征 → Query**
       - 将点云特征映射到查询空间。
    2. **Linear：点云特征 → Offset / Weight**
       - 预测图像特征的采样偏移与融合权重。
    3. **图像特征采样**
       - 在图像特征图中，根据投影位置与偏移进行局部采样。
    4. **特征融合**
       - 将采样到的图像特征与点云特征进行加权融合，输出增强后的点云特征。

    ---
    ### 输入
    Args:
        lidar_feat (torch.Tensor):
            点云特征张量，shape 为 **(N, C_l)**，
            N 表示 voxel / 点数量，C_l 表示点云特征维度。

        image_feat (torch.Tensor):
            图像特征张量，shape 为 **(B, C_i, H, W)**，
            B 为 batch size，C_i 为图像特征通道数。

        proj_points (torch.Tensor):
            点云特征对应的图像投影坐标，shape 为 **(N, 2)**，
            表示每个点在图像平面上的 (u, v) 位置。

    ---
    ### 输出
    Returns:
        torch.Tensor:
            融合后的点云特征，shape 为 **(N, C_l)**，
            每个点云特征已注入对应的图像语义信息。

    ---
    ### 模块用途
    - 在全局尺度下完成点云与图像的跨模态特征融合
    - 为后续 proposal 级与局部融合提供语义增强的点云表示
    - 属于 LoGoNet 的 GlobalFusion（GoF）阶段核心算子
    """

    def __init__(self, lidar_channels: int, image_channels: int):
        super().__init__()
        self.query_fc = nn.Linear(lidar_channels, lidar_channels)
        self.value_fc = nn.Linear(image_channels, lidar_channels)

    def forward(
        self,
        lidar_feat: torch.Tensor,
        image_feat: torch.Tensor,
        proj_points: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播过程。
        """
        # 点云特征映射到查询空间
        query = self.query_fc(lidar_feat)  # (N, C_l)

        # 将投影坐标归一化到 [-1, 1]，用于 grid_sample
        h, w = image_feat.shape[-2:]
        norm_u = proj_points[:, 0] / (w - 1) * 2 - 1
        norm_v = proj_points[:, 1] / (h - 1) * 2 - 1
        grid = torch.stack([norm_u, norm_v], dim=1).view(1, -1, 1, 2)

        # 从图像特征中采样对应位置的特征
        sampled_feat = torch.nn.functional.grid_sample(
            image_feat,
            grid,
            align_corners=True
        ).view(image_feat.shape[1], -1).t()  # (N, C_i)

        # 图像特征映射到点云特征空间
        image_emb = self.value_fc(sampled_feat)

        # 跨模态特征融合
        fused_feat = query + image_emb

        return fused_feat
