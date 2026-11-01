"""前景像素直接索引。原源码：mmdet3d/models/detectors/MSMDFusion.py，
MSMDFusionDetector.get_foreground2D，169–224 行。
"""

from __future__ import annotations

import torch
from torch.library import Library
from utils.feature_gather_by_pixel_index import feature_gather_by_pixel_index


def _impl(feats: torch.Tensor, pixels: torch.Tensor, offsets: torch.Tensor,
          lidar2img: torch.Tensor, input_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Args: feats [B*6,49,H,W]；pixels [P,3+] 为图像 x,y,depth；
    offsets [B*6+1] 保留相机顺序；lidar2img [B*6,4,4]；input_width
    为图像输入宽度。Returns: sampled [P,49] 与 score_input [P,66]。
    坐标用 W/input_width 缩放并截断为 long，随后原样高级索引；负索引按
    PyTorch 规则绕回，正向越界抛错，不裁剪。特征索引可微，像素索引不可微。
    """
    scale = feats.shape[-1] / input_width
    sampled, score_inputs = [], []
    for view in range(feats.shape[0]):
        start, end = offsets[view].item(), offsets[view + 1].item()
        pxl = pixels[start:end]
        coords = (pxl * scale).long()
        feat = feature_gather_by_pixel_index(coords[:, :2], feats[view], "long", "CHW", False)
        sensor = lidar2img[view].reshape(1, 16).repeat(feat.shape[0], 1).float()
        sampled.append(feat)
        score_inputs.append(torch.cat([feat, pxl[:, 2:3], sensor], dim=1))
    return torch.cat(sampled, dim=0), torch.cat(score_inputs, dim=0)


def _meta(feats: torch.Tensor, pixels: torch.Tensor, offsets: torch.Tensor,
          lidar2img: torch.Tensor, input_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = pixels.shape[0]
    score_dtype = torch.promote_types(torch.promote_types(feats.dtype, pixels.dtype), torch.float32)
    return (torch.empty((count, feats.shape[1]), device='meta', dtype=feats.dtype),
            torch.empty((count, feats.shape[1] + 17), device='meta', dtype=score_dtype))



_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("msmdfusion_foreground_feature_sampling", _impl)

_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("msmdfusion_foreground_feature_sampling", _meta)

def msmdfusion_foreground_feature_sampling(feats: torch.Tensor, pixels: torch.Tensor,
                                offsets: torch.Tensor, lidar2img: torch.Tensor,
                                input_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """逐相机离散采样前景特征并组装深度、投影矩阵评分输入。

    Args: feats 为 [V,C,H,W] 图像特征；pixels 为 [P,3+] (u,v,米)；
        offsets 为 [V+1]；lidar2img 为 [V,4,4]，方向 LiDAR→图像；
        input_width 是采样前图像宽度，单位像素。
    Returns: sampled [P,C] 与 score_input [P,C+17]，与特征同 device；
        score_input dtype 由拼接输入提升决定。像素坐标缩放后截断，
        不裁剪；负索引回绕，正向越界报错，索引不可微。
    """
    return torch.ops.fusionrepo.msmdfusion_foreground_feature_sampling(
        feats, pixels, offsets, lidar2img, input_width)
