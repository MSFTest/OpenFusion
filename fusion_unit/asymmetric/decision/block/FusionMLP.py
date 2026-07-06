import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    """
    FusionMLP 模块（CLOCs_LQS 融合阶段核心算子）。

    该模块对每个 (2D–3D Pair) 的融合特征进行非线性映射，用于学习 2D 检测器与 3D 检测器之间的
    关联关系。其本质是由多个 `1×1` 卷积层组成的 MLP，能够在不改变特征空间几何结构的情况下，
    对每个 pair 进行逐点特征变换，从而输出融合后的 pair-level 得分。

    本模块属于 **可学习算子**，其参数在训练过程中根据监督信号更新。

    ---
    ### 结构说明（按顺序）
    1. **Conv2d 4 → 18, kernel=1**
       - 初步映射 4 维融合特征（IoU₂D、score₂D、score₃D、distance₃D）。
    2. **ReLU 激活**
    3. **Conv2d 18 → 36, kernel=1**
       - 加强特征表达能力。
    4. **ReLU 激活**
    5. **Conv2d 36 → 36, kernel=1**
       - 非线性变换，提升 pair 语义区分能力。
    6. **ReLU 激活**
    7. **Conv2d 36 → 1, kernel=1**
       - 输出每个 pair 的融合得分。

    ---
    ### 输入
    Args:
        pair_features (torch.Tensor):
            Pair 特征张量，shape 为 **(B, 4, 1, p)**
            - B：batch size
            - 4：融合特征维度（IoU₂D、score₃D、score₂D、distance₃D）
            - 1：固定 spatial 维
            - p：pair 数量

    ---
    ### 输出
    Returns:
        torch.Tensor:
            融合后的 pair-level 得分，shape：**(B, 1, 1, p)**

    ---
    ### 模块用途
    - 为每个 (2D 检测框, 3D 检测框) 生成一个融合得分
    - 根据 MLP 学习到的 2D/3D 信息关联规则，提升最终 3D 检测结果的排序和置信度估计
    - 属于 CLOCs_LQS 的 Stage3：FusionMLP 核心部分
    """
    def __init__(self,
                 in_channels: int = 4,
                 mid_channels1: int = 18,
                 mid_channels2: int = 36,
                 mid_channels3: int = 36):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels1, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels1, mid_channels2, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels2, mid_channels3, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels3, 1, 1)
        )

    # fusion_unit/asymmetric/decision/block/FusionMLP.py 的 forward 改成这样
    def forward(self, pair_features):
        if pair_features.numel() == 0:
            return pair_features.new_zeros((0,), dtype=torch.float32)
        if pair_features.dim() == 2:
            pair_features = pair_features.unsqueeze(-1).unsqueeze(-1)  # (P,C,1,1)
        out = self.mlp(pair_features)
        return out.view(-1)
