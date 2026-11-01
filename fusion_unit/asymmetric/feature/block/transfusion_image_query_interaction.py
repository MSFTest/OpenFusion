"""输入/输出浮点张量遵循interface的float32契约；索引及可见关系见对应operator。
N7 来源：transfusion_head.py:44/726–732，forward_single:902–913、979–1004。
输入断梯度LiDAR query[B,128,P]、raw图像[B,128,H,6,W]、F3投影及网格。
输出图像query[B,128,P]和bool实际处理可见性[B,P]。图像attention可微；
每个sample/view始终重新读取同一prev_query；后视图clone写回覆盖先前结果。
同一decoder实例共享于所有视图，按原batch=1调用位置BN；state_dict键对应decoder.1。
F4通过torch.ops调用；不内嵌SMCA几何算法。last_trace只记公开节点执行顺序。
"""
import torch
from common.query_transformer_decoder_layer import QueryTransformerDecoderLayer
from common.learned_position_embedding_1d import LearnedPositionEmbedding1d
from common.transfusion_lidar_query_decoder import _initialize_decoder
from utils.transfusion_smca_constraint import transfusion_smca_constraint

class ImageQueryInteraction(QueryTransformerDecoderLayer):
    """完整原decoder；默认值来源配置195–203；不以单层注意力替代。"""
    def __init__(self,channels=128,heads=8,ffn=256,dropout=0.1,momentum=0.1):
        super().__init__(channels,heads,ffn,dropout,"relu",
                         LearnedPositionEmbedding1d(2,channels),LearnedPositionEmbedding1d(2,channels))
        _initialize_decoder(self,momentum)
        self.last_trace = []

    def forward(self,prev_query_feat,raw_img_feat,center_xy,corner_xy,visible,processed,img_pos,stride=4):
        b,c,h,v,w = raw_img_feat.shape
        img_feat = raw_img_feat.permute(0,3,1,2,4)
        img_flat = img_feat.view(b,v,c,-1)
        query_feat = torch.zeros_like(prev_query_feat)
        on_mask = torch.ones([b,prev_query_feat.shape[-1]],device=prev_query_feat.device)*-1
        self.last_trace = []
        for sample_idx in range(b):
            for view_idx in range(v):
                if not processed[sample_idx,view_idx]:
                    continue
                on_image = visible[sample_idx,view_idx]
                on_mask[sample_idx,on_image] = view_idx
                self.last_trace.append('F4')
                query_pos,attn_mask = transfusion_smca_constraint(
                    center_xy[sample_idx,view_idx],corner_xy[sample_idx,view_idx],on_image,img_pos,stride)
                query_view = prev_query_feat[sample_idx,:,on_image]
                self.last_trace.append('N7')
                query_view = super().forward(query_view[None],img_flat[sample_idx:sample_idx+1,view_idx],
                                              query_pos,img_pos,attn_mask=attn_mask)
                query_feat[sample_idx,:,on_image] = query_view.clone()
        if not self.last_trace:
            self.last_trace.append('N7:empty_views')
        return query_feat,on_mask != -1
