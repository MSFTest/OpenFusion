import torch
import torch.nn as nn


class RoIRefineHead(nn.Module):
    """
    RoIRefineHead 模块（LoGoNet 候选框回归与打分阶段算子）。

    该模块基于经过特征动态聚合（FDA）后的网格级融合特征，
    对每个候选框（RoI）进行最终的类别置信度预测与三维框参数回归。
    其核心思想是：先将 RoI 内部的网格特征进行汇聚，形成候选框级表示，
    再通过轻量级前馈网络分别完成分类与回归任务。

    本模块属于 **可学习算子**，其参数在训练过程中根据检测监督信号更新。

    ---
    ### 结构说明（按顺序）
    1. **网格特征汇聚（Mean Pooling）**
       - 将 RoI 内部 G 个网格特征聚合为单一候选框特征。
    2. **共享全连接层**
       - 对候选框级特征进行非线性映射，形成高层语义表示。
    3. **分类分支（Classification Head）**
       - 输出每个候选框的类别置信度。
    4. **回归分支（Regression Head）**
       - 输出候选框的三维框回归参数。

    ---
    ### 输入
    Args:
        roi_feat (torch.Tensor):
            RoI 内部聚合后的网格级特征，shape 为 **(K, G, C)**，
            K 为候选框数量，G 为每个候选框的网格数量，C 为特征维度。

    ---
    ### 输出
    Returns:
        cls_score (torch.Tensor):
            候选框分类得分，shape 为 **(K, num_classes)**。

        reg_offset (torch.Tensor):
            候选框回归结果，shape 为 **(K, box_dim)**，
            box_dim 通常为 7，对应 (x, y, z, dx, dy, dz, yaw)。

    ---
    ### 模块用途
    - 对融合后的候选框进行最终分类与精细位置回归
    - 提升 3D 检测结果的定位精度与置信度可靠性
    - 属于 LoGoNet 检测头（Detection Head）的最后阶段
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        box_dim: int = 7
    ):
        super().__init__()

        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
        )

        self.cls_head = nn.Linear(in_channels, num_classes)
        self.reg_head = nn.Linear(in_channels, box_dim)

    def forward(self, roi_feat: torch.Tensor):
        """
        前向传播过程。
        """
        # 1) 网格维度聚合（Mean Pooling）
        pooled_feat = roi_feat.mean(dim=1)  # (K, C)

        # 2) 共享特征变换
        feat = self.shared_fc(pooled_feat)

        # 3) 分类与回归预测
        cls_score = self.cls_head(feat)
        reg_offset = self.reg_head(feat)

        return cls_score, reg_offset
