"""为新图像adapter恢复采样梯度，不改变任务二F5公开的不可微契约。"""
import torch
from torch import nn


class _DecorationGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, batch):
        from model.PointAugmenting.data.operator_inputs import decorate_voxel_inputs
        ctx.save_for_backward(features,batch['voxels'],batch['voxels_uv'],batch['voxel_valid'])
        return decorate_voxel_inputs(batch,features)['features']

    @staticmethod
    def backward(ctx, grad_output):
        features,voxels,grid,valid=ctx.saved_tensors
        b,_,_,h,w=features.shape
        m,t=grid.shape[1],voxels.shape[1]
        packed=grad_output.new_zeros((b*m,t,64))
        packed[valid]=grad_output[:,:,5:]*voxels[:,:,-1:]
        gradient=packed.view(b,1,m,t,64).transpose(1,4).contiguous()
        volume=features.view(b,6,64,h,w).transpose(2,1)
        grad,_=torch.ops.aten.grid_sampler_3d_backward.default(
            gradient,volume,grid,0,0,False,[True,False])
        return grad.transpose(2,1),None


class TrainableImageDecorationBlock(nn.Module):
    """原voxelnet_fusion.get_img_feat前向及显式adapter反向。

    Args:
        batch: voxels[Nv,T,9]、grid[B,M,T,1,3]和bool valid[B*M]，同device；
            grid为归一化(u,v,camera)，sum(valid)=Nv，点字段为xyzrt/uvc/slot-valid。
        features: 浮点[B,6,64,H,W]，连续，相机顺序与原数据相同。
    Returns:
        原reader输入字典；features[Nv,T,69]。仅图像特征可微；几何、离散索引不可微。
        保留五维bilinear/zeros/align_corners=False以及重复点和相机边界语义。
        本block无参数和buffer；新增adapter在interface中训练。
    """
    def forward(self,batch,features):
        from model.PointAugmenting.data.operator_inputs import decorate_voxel_inputs
        if torch.is_grad_enabled() and features.requires_grad:
            value=_DecorationGradient.apply(features,batch)
            return dict(features=value,num_voxels=batch['num_points'],coors=batch['coordinates'],
                        batch_size=len(batch['num_voxels']),input_shape=batch['shape'][0])
        return decorate_voxel_inputs(batch,features)
