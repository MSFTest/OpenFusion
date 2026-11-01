from torch import nn

from common.conv2d_norm_activation import Conv2dNormActivationBlock


class AutoAlignV2ImageLateral(nn.Module):
    """原融合层内的三路独立 lateral；不属于可冻结外部图像模型。

    来源：multi_voxel_deformfusion.py::__init__ 180–193、obtain_mlvl_feats 296–301。
    Args:
        img_channels: 每层输入通道或通道序列，正式为 96；mid_channels 正式为 64。
        img_levels: 原金字塔索引顺序，正式 (0,1,2)。
        forward(img_feats): 浮点 Tensor[]，每项 [B*V,C_i,H_i,W_i]，同 dtype/device。
    Returns:
        选定各层的 [B*V,mid_channels,H_i,W_i] 列表，保持原图像/相机/空间布局。
        三路 Conv2d 3×3、stride=1、padding=1、bias=True，无 BN/激活/残差。
        初始化与无 activation 的原 MMCV ConvModule 相同：Kaiming normal fan_out、
        bias=0；无 running buffers 或 train/eval 差异，参数始终可训练。
        复用已有卷积块；原 lateral_convs.i.conv.* 键通过显式映射加载。
    """

    def __init__(self, img_channels=96, mid_channels=64, img_levels=(0, 1, 2)):
        super().__init__()
        self.img_levels = tuple(img_levels)
        channels = [img_channels] * len(self.img_levels) if isinstance(img_channels, int) else list(img_channels)
        assert len(channels) == len(self.img_levels)
        self.lateral_convs = nn.ModuleList()
        for channel in channels:
            block = Conv2dNormActivationBlock(channel, mid_channels, 3, 1,
                                              use_bn=False, activation=False, inplace=False)
            nn.init.kaiming_normal_(block.block[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(block.block[0].bias, 0.)
            self.lateral_convs.append(block)

    def forward(self, img_feats):
        return [conv(img_feats[level]) for level, conv in zip(self.img_levels, self.lateral_convs)]

    def map_original_state_dict(self, state_dict):
        """映射原 lateral 子状态，不复制 Tensor；漏键/未知键明确失败，shape 由严格加载检查。"""
        mapped = {key.replace('.conv.', '.block.0.'): value for key, value in state_dict.items()}
        if set(mapped) != set(self.state_dict()):
            raise KeyError('原 lateral 参数键缺失或含未知键')
        return mapped
