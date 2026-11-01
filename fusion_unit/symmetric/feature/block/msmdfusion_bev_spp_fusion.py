"""BEV SPP 融合。原源码：mmdet3d/models/detectors/MSMDFusion.py，
SPPModule.__init__ 47–80、forward 82–90、MSMDFusionDetector.
extract_pts_feat 440–445 行。
"""

import torch
from torch import nn
from fusion_unit.symmetric.feature.ops.feature_concatenation import feature_concatenation


class BEVSPPFusion(nn.Module):
    """输入纯 LiDAR BEV 256 与多模态 BEV 384 拼接成 [B,640,H,W]。

    四分支均输出 256：1x1、3x3、3x3 dilation6、3x3 dilation12；
    各卷积 bias=False、BN eps=0.001 momentum=0.01、ReLU；拼接
    1024 后 1x1 融合为 256。state_dict 键与原 SPPModule 一致。

    Args: forward 的 pure_lidar_bev 为 [B,256,H,W]，multimodal_bev
        为 [B,384,H,W]；二者使用同一 LiDAR BEV 米制网格。
    Returns: [B,256,H,W] 融合特征，dtype/device 跟随输入。
    """

    def __init__(self):
        super().__init__()

        def block(kernel, padding, dilation=1):
            return nn.Sequential(
                nn.Conv2d(640, 256, kernel, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(256, eps=0.001, momentum=0.01), nn.ReLU())

        self.conv1x1 = block(1, 0)
        self.conv3x3 = block(3, 1)
        self.dilated_conv3x3_rate6 = block(3, 6, 6)
        self.dilated_conv3x3_rate12 = block(3, 12, 12)
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01), nn.ReLU())

    def forward(self, pure_lidar_bev: torch.Tensor, multimodal_bev: torch.Tensor) -> torch.Tensor:
        """原 extract_pts_feat 445 行与 SPPModule.forward 82–90 行。"""
        x = feature_concatenation([pure_lidar_bev, multimodal_bev], 1)
        return self.fuse(feature_concatenation([
            self.conv1x1(x), self.conv3x3(x),
            self.dilated_conv3x3_rate6(x), self.dilated_conv3x3_rate12(x)], 1))
