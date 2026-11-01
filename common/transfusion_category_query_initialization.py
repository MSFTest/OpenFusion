"""N4 来源：transfusion_head.py:689、forward_single:865–873（起始797）。
输入原LiDAR tokens[B,C,N]、int64类别/索引[B,P]、BEV位置[B,N,2]。
输出query[B,C,P]、位置[B,P,2]；gather/类别编码可微，类别与索引不可微。
保留class_encoding.*原键与原位 +=；默认冻结来自tools/train.py:194。
"""
from torch import nn
from common.bevdet_proposal_class_encoding import ProposalClassEncoding
import torch.nn.functional as F

class CategoryQueryInitialization(nn.Module):
    """类别10/通道128来自transfusion_nusc_voxel_LC.py:193–194。"""
    def __init__(self,classes=10,channels=128):
        super().__init__()
        self.num_classes = classes
        self.class_encoding = ProposalClassEncoding(channels, classes).class_encoding

    def forward(self,lidar_tokens,labels,indices,bev_pos):
        feat = lidar_tokens.gather(index=indices[:,None,:].expand(-1,lidar_tokens.shape[1],-1),dim=-1)
        one_hot = F.one_hot(labels,num_classes=self.num_classes).permute(0,2,1)
        feat += self.class_encoding(one_hot.float())
        pos = bev_pos.gather(index=indices[:,None,:].permute(0,2,1).expand(-1,-1,bev_pos.shape[-1]),dim=1)
        return feat,pos
