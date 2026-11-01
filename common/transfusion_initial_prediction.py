"""TransFusion 初始六字段预测；源transfusion_head.py:507/705/890–897。"""
from torch import nn
from common.deepinteraction_prediction_head import PredictionHead


def _initialize_prediction(module, momentum):
    # 原 FFN 构造时 ConvModule 初始化第一卷积；未调用 FFN.init_weights，heatmap偏置不能另设-2.19。
    for layer in module.modules():
        if isinstance(layer, nn.Conv1d) and layer.bias is None:
            nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_out', nonlinearity='relu')
        elif isinstance(layer, nn.BatchNorm1d):
            layer.momentum = momentum


class InitialPrediction(PredictionHead):
    """六分支每个Conv1d(C,64,1,biasFalse)-BN-ReLU-Conv1d(64,D,1)，独立参数。
    Args: in_channels128/classes10/momentum0.1；forward query_feat[B,128,P]、query_pos[B,P,2]。
    Returns: (dict, detached_center[B,P,2], detached_query[B,128,P])；六字段分别
    center/height/dim/rot/vel/heatmap 的 D=2/1/3/2/2/10，dtype/device同输入。
    center为BEV格坐标，height为LiDAR几何中心z米，dim为log(dx,dy,dz)，rot为sin/cos旧yaw，
    vel为LiDAR vx/vy米/秒；中心残差在预测后相加。train/eval与BN buffer保留。
    remap_original_state_dict映射原FFN键到复用公共预测头，strict加载验证；参数无冻结副作用。
    """
    def __init__(self, in_channels=128, classes=10, momentum=0.1):
        super().__init__(in_channels, classes)
        _initialize_prediction(self, momentum)

    def forward(self, query_feat, query_pos):
        result = super().forward(query_feat)
        result['center'] = result['center'] + query_pos.permute(0, 2, 1)
        return result, result['center'].detach().clone().permute(0, 2, 1), query_feat.detach().clone()
