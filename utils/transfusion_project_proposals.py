"""原六视图 proposal 投影准备，源 transfusion_head.py:914–985，保留非对称历史坐标约定。
Args: predictions Dict 六字段[B,D,P]、query_pos[B,P,2]；lidar2img 等为每样本 Tensor[]；
矩阵[6,4,4]为LiDAR米制坐标到图像齐次坐标，scales/crops[2]、image/padded_shapes[2]；
三维逆增强参数及 JSON flows；out_size、voxel_x/y 和pc_x/y为格到米参数。
Returns: centers[B,6,P,2]、corners[B,6,P,8,2] 像素坐标，以及 bool visible[B,6,P]、
processed[B,6]。浮点输出源 dtype 提升，设备同 query；无梯度；接受非连续字段。
box 使用旧LiDAR底中心、dx/dy/dz、旧旋转方向（复用角点时只对临时yaw取反）。
P=0 按源空框assert失败；P=1 固定输出维且 processed=False；同像素不去重。
中心两轴沿用 voxel_x/pc_x；B==1 跳过逆3D增强；深度仅clamp1e-5，无正深度检查；
resize→crop→flip，翻转用未pad宽，严格0<x<padW/0<y<padH，有效数<=1跳过decoder。
Meta/Fake只表达固定输出；投影设备/dtype不匹配、奇异旋转等原错误保留。
"""
import torch
from torch.library import Library
from utils import operator_registry as _registry
import json
from utils.deepinteraction_decode_boxes import deepinteraction_decode_boxes
from utils.deepinteraction_lidar_box_corners import deepinteraction_lidar_box_corners
from utils.batched_homogeneous_transform_and_normalize import batched_homogeneous_transform_and_normalize
from utils.transfusion_reverse_lidar_transform import transfusion_reverse_lidar_transform


def _impl(predictions, query_pos, lidar2img, scales, crops, image_shapes, padded_shapes, flips, rotations, pcd_scales, translations, hflips, vflips, flows, out_size, voxel_x, voxel_y, pc_x, pc_y):
    b,p = query_pos.shape[:2]
    metric = query_pos.permute(0,2,1)*out_size*voxel_x+pc_x
    pos3d = torch.cat([metric,predictions['height']],dim=1).detach().clone()
    velocity = predictions.get('vel')
    if velocity is None:
        velocity = predictions['center'].new_empty((b, 0, p))
    boxes = deepinteraction_decode_boxes(
        predictions['center'].detach().clone(), predictions['height'].detach().clone(),
        predictions['dim'].detach().clone(), predictions['rot'].detach().clone(),
        velocity.detach().clone(), query_pos.new_tensor([pc_x, pc_y, 0, 0, 0, 0]),
        query_pos.new_tensor([voxel_x, voxel_y, 0]), out_size)
    all_centers,all_corners,all_visible,all_processed = [],[],[],[]
    for sample_idx in range(b):
        old_boxes = boxes[sample_idx, :, :7].to(torch.float32).clone()
        assert len(old_boxes) != 0
        old_boxes[:, 6] = -old_boxes[:, 6]
        corners = deepinteraction_lidar_box_corners(old_boxes)
        points = torch.cat([pos3d[sample_idx],corners.permute(2,0,1).view(3,-1)],dim=-1).T
        if b != 1:
            points = transfusion_reverse_lidar_transform(points,rotations[sample_idx],pcd_scales[sample_idx],
                      translations[sample_idx],hflips[sample_idx],vflips[sample_idx],
                      json.loads(flows[sample_idx])).detach()
        centers_v,corners_v,visible_v,processed_v = [],[],[],[]
        for view_idx in range(6):  # H:951；最终配置num_views=6，不遍历额外metadata视图。
            pts2 = batched_homogeneous_transform_and_normalize(points, lidar2img[sample_idx][view_idx], False)[0]
            pts2[:,2] = torch.clamp(pts2[:,2],min=1e-5)
            pts2[:,0] /= pts2[:,2]
            pts2[:,1] /= pts2[:,2]
            coors = pts2[:,:2]*scales[sample_idx]
            coors -= crops[sample_idx]
            x,y = torch.split(coors,1,dim=1)
            if flips[sample_idx]:
                x = image_shapes[sample_idx][1]-x
            cx,xx = x[:p],x[p:]
            cy,yy = y[:p],y[p:]
            corner_xy = torch.cat([xx.reshape(p,8,1),yy.reshape(p,8,1)],dim=-1)
            h,w = padded_shapes[sample_idx]
            visible = ((cx>0)*(cx<w)*(cy>0)*(cy<h)).squeeze(-1)
            centers_v.append(torch.cat([cx,cy],dim=-1))
            corners_v.append(corner_xy)
            visible_v.append(visible)
            processed_v.append(visible.sum()>1)
        all_centers.append(torch.stack(centers_v))
        all_corners.append(torch.stack(corners_v))
        all_visible.append(torch.stack(visible_v))
        all_processed.append(torch.stack(processed_v))
    return tuple(torch.stack(x) for x in (all_centers,all_corners,all_visible,all_processed))


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("transfusion_project_proposals", _impl)


def _meta(predictions, query_pos, lidar2img, scales, crops, image_shapes, padded_shapes, flips, rotations, pcd_scales, translations, hflips, vflips, flows, out_size, voxel_x, voxel_y, pc_x, pc_y):
    b, p = query_pos.shape[:2]
    assert p != 0
    if b == 0:
        raise RuntimeError("原六视图投影不能stack空batch")
    dtype = torch.promote_types(query_pos.dtype, predictions['height'].dtype)
    dtype = torch.promote_types(dtype, torch.float32)
    point_dtype = dtype if b == 1 else torch.float32
    if any(matrix.dtype != point_dtype for matrix in lidar2img):
        raise RuntimeError("原投影矩阵与点dtype必须一致")
    dtype = point_dtype
    for sample_idx in range(b):
        if lidar2img[sample_idx].shape[0] < 6:
            raise IndexError("原投影要求六个相机矩阵")
        if b != 1:
            # 用同一逆增强Meta契约验证dtype/flow/零scale，不能漏过未执行的值依赖分支错误。
            transfusion_reverse_lidar_transform(
                torch.empty((p*9, 3), dtype=point_dtype, device='meta'),
                rotations[sample_idx], pcd_scales[sample_idx], translations[sample_idx],
                hflips[sample_idx], vflips[sample_idx], json.loads(flows[sample_idx]))
        dtype = torch.promote_types(dtype, scales[sample_idx].dtype)
    return (torch.empty((b,6,p,2),dtype=dtype,device="meta"),
            torch.empty((b,6,p,8,2),dtype=dtype,device="meta"),
            torch.empty((b,6,p),dtype=torch.bool,device="meta"),
            torch.empty((b,6),dtype=torch.bool,device="meta"))


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("transfusion_project_proposals", _meta)


def transfusion_project_proposals(predictions,query_pos,img_metas,out_size=8,voxel_size=(0.075,0.075),pc_range=(-54.,-54.)):
    """投影 adapter：predictions 六字段[B,D,P]、query_pos[B,P,2] 和每样本原 metadata。
    Returns: centers[B,6,P,2]、corners[B,6,P,8,2] 像素坐标，visible bool[B,6,P]、
    processed bool[B,6]（有效query>1）；输出不含梯度，浮点 dtype 由几何计算提升。
    voxel_size 米/体素，pc_range 为LiDAR最小(x,y)米，out_size 为BEV降采样倍数。
    调用 torch.ops 时 metadata 展为 Tensor[]/标量；B==1 不读取三维增强信息。
    """
    # H:915/932–938：metadata沿用cat(center,height)后的dtype，而不是center单独的dtype。
    anchor = query_pos.new_empty((),dtype=torch.promote_types(query_pos.dtype,predictions['height'].dtype))
    tensors = lambda key,default: [anchor.new_tensor(m.get(key,default)) for m in img_metas]
    l2i = [anchor.new_tensor(m['lidar2img']) for m in img_metas]
    scales = [anchor.new_tensor(m['scale_factor'][:2] if 'scale_factor' in m else [1.,1.]) for m in img_metas]
    shapes = [anchor.new_tensor(m['img_shape'][:2]) for m in img_metas]
    pads = [anchor.new_tensor(m['input_shape'][:2]) for m in img_metas]
    # H:946–949：B==1根本不读取三维增强metadata，不能让未执行分支产生额外错误。
    transform_metas = img_metas if query_pos.shape[0] != 1 else [{} for _ in img_metas]
    rotations = [torch.tensor(m['pcd_rotation'],dtype=anchor.dtype,device=anchor.device)
                 if 'pcd_rotation' in m else torch.eye(3,dtype=anchor.dtype,device=anchor.device) for m in transform_metas]
    return torch.ops.fusionrepo.transfusion_project_proposals(predictions,query_pos,l2i,scales,
        tensors('img_crop_offset',0),shapes,pads,[bool(m.get('flip',False)) for m in img_metas],
        rotations,[float(m.get('pcd_scale_factor',1.)) for m in transform_metas],
        [anchor.new_tensor(m.get('pcd_trans',[0.,0.,0.])) for m in transform_metas],
        [bool(m.get('pcd_horizontal_flip',False)) for m in transform_metas],
        [bool(m.get('pcd_vertical_flip',False)) for m in transform_metas],
        # coord_transform.py:53/85：完整保留flow元素；分隔符不能把非法单个标签误当多个有效标签。
        [json.dumps(m.get('transformation_3d_flow',[])) for m in transform_metas],out_size,
        float(voxel_size[0]),float(voxel_size[1]),float(pc_range[0]),float(pc_range[1]))
