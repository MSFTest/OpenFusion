"""N1 图像高度压缩。来源：transfusion_head.py:741、forward_single:819–823。
Args: [B*6,C,H,W] float32 图像特征；B 为样本数，视图顺序来自原 metadata。
Returns: [B,C,6W] 列特征及 [B,C,H,6,W] 原 raw_img_feat；max 与 fc 均可微。
保留高度 max 的并列最大值反向语义；fc 的 state_dict 保持 fc.0.* 原键。
"""
import torch
from torch import nn
from common.indexed_max_reduce import indexed_max_reduce

class ImageColumnCompression(nn.Module):
    """Conv1d C→C,k=1；C=128 来自 transfusion_nusc_voxel_LC.py:193。"""
    def __init__(self, channels=128, num_views=6):
        super().__init__()
        self.num_views = num_views
        self.fc = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, img_feat, batch_size):
        h,w,c = img_feat.shape[-2],img_feat.shape[-1],img_feat.shape[1]
        raw = img_feat.view(batch_size,self.num_views,c,h,w).permute(0,2,3,1,4)
        feat = raw.reshape(batch_size,c,h,w*self.num_views)
        collapsed, _ = indexed_max_reduce(feat, 2)
        return self.fc(collapsed).view(batch_size,c,w*self.num_views), raw
