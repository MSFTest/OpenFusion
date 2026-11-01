"""多视角稀疏深度图。原源码：mmdet3d/models/detectors/MSMDFusion.py，
MSMDFusionDetector.depth_aware_channel_compression，335 行，具体写入见 337–358 行。
"""

import torch
from torch.library import Library


def _impl(real_pixels: torch.Tensor, offsets: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Args: real_pixels [P,3+] 为原图像像素 x,y,depth；offsets [B*6+1]。

    Returns: 默认浮点 dtype 的 [B*6,1,H,W]，图像像素系。无可学习参数；输入深度源自
    NumPy，不参与反传。逐视角使用原始 index_put_，重复像素与越界行为保留。
    视角必须依样本顺序排列；H/W 取首个 img_meta 的 pad_shape。
    """
    views = offsets.numel() - 1
    canvas = torch.zeros(views, height, width, device=real_pixels.device)
    for view in range(views):
        pixels = real_pixels[offsets[view].item():offsets[view + 1].item()]
        coors = pixels[:, :2].long()
        canvas[view].index_put_((coors[:, 1], coors[:, 0]), pixels[:, 2])
    return canvas.view(views, 1, height, width)


def _meta(real_pixels: torch.Tensor, offsets: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return torch.empty((offsets.numel() - 1, 1, height, width), dtype=torch.get_default_dtype(), device='meta')



_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("msmdfusion_sparse_depth_map", _impl)

_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("msmdfusion_sparse_depth_map", _meta)

def msmdfusion_sparse_depth_map(real_pixels: torch.Tensor, offsets: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """从各相机真实前景像素构造稀疏深度图。

    Args: real_pixels 为 [P,3+] (图像 u,v,深度米)，offsets 为
        int64 [V+1] 相机切片边界；height、width 为原图像画布尺寸。
    Returns: 与输入同 device、默认浮点 dtype 的 [V,1,H,W]。重复像素沿原
        index_put_ 写入，越界仍按 PyTorch 索引规则报错；无可微输入。
    """
    return torch.ops.fusionrepo.msmdfusion_sparse_depth_map(real_pixels, offsets, height, width)
