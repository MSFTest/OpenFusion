"""TransFusion 图像列对 BEV 的 cross-only 交互；源transfusion_head.py:733–740/831–832。"""
from common.query_transformer_decoder_layer import QueryTransformerDecoderLayer
from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.transfusion_lidar_query_decoder import _initialize_decoder


class ImageToBEVAttention(QueryTransformerDecoderLayer):
    """单视图完整decoder，调用者为六视图分别构造独立实例并顺序更新BEV。
    Args: channels128/heads8/ffn256/dropout0.1/momentum0.1；forward为
    query[B,C,Nbev], key[B,C,W], query_pos[B,Nbev,2], key_pos[1,W,2]。
    Returns: 更新BEV[B,C,Nbev]，dtype/device跟随输入；两路位置均为特征格(x,y)。
    保留未执行的norm1状态、cross-attention的Q/K/V位置、FFN/dropout/post-LN和BN。
    原decoder[2+view]的局部state_dict可直接strict加载；初始化按原xavier_uniform。
    """
    def __init__(self, channels=128, heads=8, ffn=256, dropout=0.1, momentum=0.1):
        super().__init__(channels, heads, ffn, dropout, 'relu',
                         LearnedPositionEmbedding1d(2, channels),
                         LearnedPositionEmbedding1d(2, channels), cross_only=True)
        _initialize_decoder(self, momentum)
