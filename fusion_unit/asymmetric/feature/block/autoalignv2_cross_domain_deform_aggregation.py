import torch
from torch import nn

from common.linear_projection import linear_projection
from common.autoalignv2_deform_attention_backend import autoalignv2_deform_attention_backend
from utils.autoalignv2_restore_lidar_coordinates import autoalignv2_restore_lidar_coordinates
from utils.autoalignv2_project_voxels_to_image import autoalignv2_project_voxels_to_image
from utils.autoalignv2_normalize_and_assign_camera import autoalignv2_normalize_and_assign_camera
from utils.autoalignv2_sample_reference_features import autoalignv2_sample_reference_features
from fusion_unit.asymmetric.feature.ops.autoalignv2_initialize_cross_domain_query import autoalignv2_initialize_cross_domain_query


class AutoAlignV2CrossDomainDeformAggregation(nn.Module):
    """原 Cross-Domain DeformCAFA 采样聚合，不含外围 lateral/输出 BN。

    来源：multi_voxel_deformfusion.py::__init__ 220–229、sample_single 329–462。
    Args:
        mid_channels/num_levels/num_heads/num_points: 默认 64/3/4/8，与原提取一致。
        fix_offset: 原构造默认 True；正式配置显式 False，仅控制 offset 参数梯度。
        im2col_step: 默认 64；backend: auto/pytorch/mmcv，不作依赖失败回退。
        forward(img_feats, centers, voxel_features, img_meta, trace=None, batch_index=0):
            图像各层 [V,64,H_l,W_l]，增强 LiDAR XYZ 米制 centers [M,3]，voxel
            特征 [M,64]，正式均同设备 float32；metadata 含 LiDAR→像素矩阵及增强记录。
    Returns:
        同设备 float32 [M,64] 图像聚合表示；首个可见相机占有，未分配行零值，
        重复坐标保持独立行。query 的 voxel 分支 detach，图像/value 路径及参数可微。
        原参考坐标直接加 offset/[W,H]，不修正源码坐标约定；输出加 reference 残差。
        保留三个 Linear 的默认构造初始化和 state_dict 键，无额外 output_proj、
        token FC、norm、position embedding、FFN 或 dropout。无内部相机状态；
        train/eval 算法相同。trace 仅记录实际张量，不参与计算。Meta 全块受 F05 动态 Q 限制。
    """

    def __init__(self, mid_channels=64, num_levels=3, num_heads=4, num_points=8,
                 fix_offset=True, im2col_step=64, backend='auto'):
        super().__init__()
        self.mid_channels = mid_channels
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.fix_offset = fix_offset
        self.im2col_step = im2col_step
        self.backend = backend
        self.deform_sampling_offsets = nn.Linear(mid_channels, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(mid_channels, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(mid_channels, mid_channels)
        if fix_offset:
            self.deform_sampling_offsets.requires_grad_(False)

    def forward(self, img_feats, centers, voxel_features, img_meta, trace=None, batch_index=0):
        scale = centers.new_tensor(img_meta['scale_factor'][:2]) if 'scale_factor' in img_meta else centers.new_tensor(1.)
        crop = centers.new_tensor(img_meta['img_crop_offset']) if 'img_crop_offset' in img_meta else centers.new_tensor(0.)
        rotation = torch.tensor(img_meta['pcd_rotation'], dtype=centers.dtype, device=centers.device) if 'pcd_rotation' in img_meta else torch.eye(3, dtype=centers.dtype, device=centers.device)
        translation = torch.tensor(img_meta['pcd_trans'], dtype=centers.dtype, device=centers.device) if 'pcd_trans' in img_meta else centers.new_zeros(3)
        already = centers.new_zeros(centers.shape[0], dtype=torch.bool)
        result = centers.new_zeros((centers.shape[0], self.mid_channels))
        for camera in range(img_feats[0].shape[0]):
            restored = autoalignv2_restore_lidar_coordinates(
                centers, rotation, float(img_meta.get('pcd_scale_factor', 1.)), translation,
                bool(img_meta.get('pcd_horizontal_flip', False)), bool(img_meta.get('pcd_vertical_flip', False)),
                img_meta.get('transformation_3d_flow', []))
            projection = centers.new_tensor(img_meta['lidar2img'][camera])
            uvd = autoalignv2_project_voxels_to_image(restored, projection, scale, crop)
            height, width = img_meta['input_shape'][:2]
            grid, valid, assigned, already = autoalignv2_normalize_and_assign_camera(uvd, height, width, already)
            camera_feats = [feature[camera:camera + 1] for feature in img_feats]
            reference = autoalignv2_sample_reference_features(camera_feats, grid, assigned)
            query = autoalignv2_initialize_cross_domain_query(voxel_features, reference, assigned)
            if trace is not None:
                for name, tensor in [('F02', restored), ('F03', uvd), ('F04', grid), ('F05', reference), ('F06', query)]:
                    trace.append(dict(id=name, scene=batch_index, camera=camera, tensor=tensor))
            valid_grid = grid[assigned].unsqueeze(0).unsqueeze(0)
            valid_grid = valid_grid.permute(0, 2, 1, 3).repeat(1, 1, self.num_levels, 1)
            flattened, spatial = [], []
            for feature in camera_feats:
                batch, channels, h, w = feature.shape
                spatial.append((h, w))
                flattened.append(feature.view(batch, channels, h, w).flatten(2).transpose(1, 2))
            value = torch.cat(flattened, 1)
            shapes = torch.as_tensor(spatial, dtype=torch.long, device=value.device)
            starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
            count = query.shape[1]
            offsets = linear_projection(query, self.deform_sampling_offsets.weight, self.deform_sampling_offsets.bias).view(
                1, count, self.num_heads, self.num_levels, self.num_points, 2)
            weights = linear_projection(query, self.attention_weights.weight, self.attention_weights.bias).view(
                1, count, self.num_heads, self.num_levels * self.num_points).softmax(-1)
            value = linear_projection(value, self.value_proj.weight, self.value_proj.bias)
            value = value.view(1, value.shape[1], self.num_heads, -1)
            normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
            locations = valid_grid[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
            output = autoalignv2_deform_attention_backend(value, shapes, starts, locations, weights,
                                                         self.im2col_step, self.backend) + reference
            result[assigned] = output.squeeze(0)
            if trace is not None:
                trace.append(dict(id='N01', scene=batch_index, camera=camera, tensor=output,
                                  offsets=offsets, weights=weights, locations=locations, assigned=assigned))
        return result
