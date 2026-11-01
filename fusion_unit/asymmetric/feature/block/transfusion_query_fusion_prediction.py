"""TransFusion 图像/LiDAR query拼接及最终六字段预测；源transfusion_head.py:743/1005–1006。"""
from common.deepinteraction_prediction_head import PredictionHead
from common.transfusion_initial_prediction import _initialize_prediction
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class QueryFusionPrediction(PredictionHead):
    """保留拼接顺序[image_query, prev_lidar_query]，完整六分支两层预测头。
    Args: in_channels256/classes10/momentum0.1；forward图像query[B,128,P]、位置[B,P,2]、
    prev_query_feat[B,128,P]（源调用者已detach/clone）。Returns: 六字段dict，每个[B,D,P]，
    D=2/1/3/2/2/10，dtype/device同输入；位置及box字段约定同InitialPrediction。
    不替调用者偷偷detach；参数与初始头独立，BN/train/eval/初始化保留；
    原prediction_heads.1局部state_dict通过继承的remap_original_state_dict strict映射。
    """
    def __init__(self, in_channels=256, classes=10, momentum=0.1):
        super().__init__(in_channels, classes)
        _initialize_prediction(self, momentum)

    def forward(self, query_feat, query_pos, prev_query_feat):
        result = super().forward(feature_concatenation([query_feat, prev_query_feat], 1))
        result['center'] = result['center'] + query_pos.permute(0, 2, 1)
        return result
