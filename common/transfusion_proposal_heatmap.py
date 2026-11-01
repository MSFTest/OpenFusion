"""TransFusion 两层 dense proposal heatmap；源transfusion_head.py:669–688/724/839–842。"""
import torch
from torch import nn
from common.conv2d_norm_activation import Conv2dNormActivationBlock


class ProposalHeatmap(nn.Sequential):
    """Conv2d(C,C,3,pad1)-BN-ReLU-Conv2d(C,classes,3,pad1)，不含sigmoid/NMS。
    Args: channels128/classes10/momentum0.1/bias='auto'为原config和head默认；
    输入浮点[B,C,H,W]，Returns 同device/dtype logits[B,classes,H,W]。
    参数/BN buffer和train/eval保持；第一卷积Kaiming normal fan_out/ReLU，BN1/0，
    最后卷积沿用nn.Conv2d初始化。图像head应由LiDAR head deepcopy以保留初值相同而存储独立。
    新键0.block.*，remap_original_state_dict显式映射源0.conv/0.bn/1.*，不丢任何权重。
    """
    def __init__(self, channels=128, classes=10, momentum=0.1, bias='auto'):
        block = Conv2dNormActivationBlock(channels, channels, 3, 1)
        conv, bn = block.block[0], block.block[1]
        if bias is True:
            conv.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_out', nonlinearity='relu')
        bn.momentum = momentum
        super().__init__(block, nn.Conv2d(channels, classes, 3, padding=1, bias=bool(bias)))

    @staticmethod
    def remap_original_state_dict(state_dict):
        return {key.replace('0.conv.', '0.block.0.').replace('0.bn.', '0.block.1.'): value
                for key, value in state_dict.items()}
