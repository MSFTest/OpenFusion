"""TransFusion 原 LiDAR decoder：transfusion_head.py:44/695–702/882–887。"""
from torch import nn
from common.query_transformer_decoder_layer import QueryTransformerDecoderLayer
from common.learned_position_embedding_1d import LearnedPositionEmbedding1d


def _initialize_decoder(module, momentum):
    # 原 head.init_weights:769 在所有 decoder 构造后执行；独立算子执行相同初始化。
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    for layer in module.modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            layer.momentum = momentum


class LiDARQueryDecoder(QueryTransformerDecoderLayer):
    """完整 self-attention→cross-attention→FFN，保留 position Q/K/V、三组 post-LN。
    Args: channels128、heads8、ffn256、dropout0.1、momentum0.1 为实际config默认值。
    forward(query[B,C,P], key[B,C,N], query_pos[B,P,2], key_pos[B,N,2], attn_mask=None)。
    Returns: [B,C,P]，dtype/device跟随输入；位置为BEV格坐标(x,y)，不做米制转换。
    原 state_dict 直接 strict 加载；可学习位置Conv1d-BN-ReLU-Conv1d、norm1/2/3状态均保留。
    初始化为原xavier_uniform，BN momentum可配置；train/eval保留BN/dropout；空query按原torch行为。
    冻结由任务三调用者管理，不通过删参数改变算子。
    """
    def __init__(self, channels=128, heads=8, ffn=256, dropout=0.1, momentum=0.1):
        super().__init__(channels, heads, ffn, dropout, 'relu',
                         LearnedPositionEmbedding1d(2, channels),
                         LearnedPositionEmbedding1d(2, channels))
        _initialize_decoder(self, momentum)
