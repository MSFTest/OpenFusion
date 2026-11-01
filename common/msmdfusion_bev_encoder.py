"""融合后 BEV 编码。原源码：mmdet3d/models/backbones/second.py，
SECOND.__init__ 21–62、forward 73–86 行；mmdet3d/models/necks/
second_fpn.py，SECONDFPN.__init__ 26–65、forward 75–92 行；
配置 configs/MSMDFusion_nusc_voxel_LC.py 191–206 行。
"""

from __future__ import annotations

import torch
from torch import nn


class _SECOND(nn.Module):
    """原 SECOND 两级，输入 256，输出 128/256。"""

    def __init__(self):
        super().__init__()
        blocks = []
        for in_channels, out_channels, stride in ((256, 128, 1), (128, 256, 2)):
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                      nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01), nn.ReLU(inplace=True)]
            for _ in range(5):
                layers.extend((nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                               nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01), nn.ReLU(inplace=True)))
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """原 SECOND.forward 73–86 行。"""
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return tuple(outputs)


class _SECONDFPN(nn.Module):
    """原 SECONDFPN：一级 1x1 Conv，二级 2x2 ConvTranspose，上采样后拼接。"""

    def __init__(self):
        super().__init__()
        self.deblocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 256, 1, stride=1, bias=False),
                          nn.BatchNorm2d(256, eps=0.001, momentum=0.01), nn.ReLU(inplace=True)),
            nn.Sequential(nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
                          nn.BatchNorm2d(256, eps=0.001, momentum=0.01), nn.ReLU(inplace=True)),
        ])

    def forward(self, x):
        """原 SECONDFPN.forward 75–92 行；返回单元素列表。"""
        ups = [self.deblocks[i](x[i]) for i in range(2)]
        return [torch.cat(ups, dim=1)]


class BEVEncoder(nn.Module):
    """原 detector 的 pts_backbone+pts_neck，输出 [B,512,H,W]。

    内部名字保留 pts_backbone/pts_neck，state_dict 可直接映射原检测器
    同名子树；不新建参数化节点。此两级编码由阶段 B 增补为 N7。

    Args: forward 的 bev 为 [B,256,H,W] LiDAR BEV 特征。
    Returns: 单元素列表，元素为 [B,512,H,W]；卷积、BN 与 train/eval
        状态完全沿用原 SECOND 和 SECONDFPN。
    """

    def __init__(self):
        super().__init__()
        self.pts_backbone = _SECOND()
        self.pts_neck = _SECONDFPN()

    def forward(self, bev: torch.Tensor) -> list[torch.Tensor]:
        """原 MSMDFusionDetector.extract_pts_feat 447–452 行。"""
        return self.pts_neck(self.pts_backbone(bev))
