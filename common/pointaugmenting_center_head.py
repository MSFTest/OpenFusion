"""固定六任务CenterHead。Args: [B,512,H,W]浮点BEV。Returns: 六个dict，reg/height/dim/vel/rot/hm通道为2/1/3/2/2/Ctask；BN状态和初始化保留。无DCN；loss和decode集中在唯一build文件。
原源码：det3d/models/bbox_heads/center_head.py。参数、buffer及计算顺序按原节点迁移。
"""
import logging,copy
import torch
from torch import nn
from common._pointaugmenting_layers import Sequential,kaiming_init
class SepHead(nn.Module):

    """原center_head.py::SepHead；Args为[B,64,H,W]浮点BEV，Returns为各预测分支字典。保留两层卷积、BN、激活、hm偏置及回归初始化。"""
    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, bn=False, init_bias=-2.19, **kwargs):
        super(SepHead, self).__init__(**kwargs)
        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]
            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_channels, head_conv, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())
            fc.add(nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict

class CenterHead(nn.Module):

    """原center_head.py::CenterHead；Args为[B,512,H,W]，Returns为六任务reg/height/dim/vel/rot/hm字典。同dtype/device；类别和10字段编码来自配置，BN训练状态保留。"""
    def __init__(self, in_channels=[128], tasks=[], dataset='nuscenes', weight=0.25, code_weights=[], common_heads=dict(), logger=None, init_bias=-2.19, share_conv_channel=64, num_hm_conv=2, dcn_head=False):
        if dcn_head:
            raise ValueError('正式配置要求 dcn_head=False')
        super(CenterHead, self).__init__()
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight
        self.dataset = dataset
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.use_direction_classifier = False
        if not logger:
            logger = logging.getLogger('CenterHead')
        self.logger = logger
        logger.info(f'num_classes: {num_classes}')
        self.shared_conv = nn.Sequential(nn.Conv2d(in_channels, share_conv_channel, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(share_conv_channel), nn.ReLU(inplace=True))
        self.tasks = nn.ModuleList()
        print('Use HM Bias: ', init_bias)
        if dcn_head:
            print('Use Deformable Convolution in the CenterHead!')
        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3))
            else:
                self.tasks.append(DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3))
        logger.info('Finish CenterHead Initialization')

    def forward(self, x, *kwargs):
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts

