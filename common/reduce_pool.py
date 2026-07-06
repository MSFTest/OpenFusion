import torch
from torch.library import Library

lib = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")

def reduce_pool_impl(x: torch.Tensor,
                     dim: int = 1,
                     mode: str = "max",
                     keepdim: bool = False):

    # ===== 关键修复 =====
    if x.size(dim) == 0:
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)

        return torch.full(
            shape,
            -1e7,
            device=x.device,
            dtype=x.dtype
        )

    if mode == "max":
        return torch.amax(x, dim=dim, keepdim=keepdim)

    if mode == "sum":
        return torch.sum(x, dim=dim, keepdim=keepdim)

    if mode == "mean":
        return torch.mean(x, dim=dim, keepdim=keepdim)

    raise RuntimeError(f"Unsupported reduce mode: {mode}")

lib.impl("reduce_pool", reduce_pool_impl)


# ============================================================
# wrapper
# ============================================================

def reduce_pool(
        x: torch.Tensor,
        mask: torch.Tensor,
        mode: str = "max"):
    """
    Wrapper for torch.ops.fusionrepo.reduce_pool

    Args:
        x: (B, M, N, C) grouped features
        mask: (B, M, N) boolean mask indicating valid points
        mode: "max" | "mean" | "sum"

    Returns:
        pooled: (B, M, C)
    """
    return torch.ops.fusionrepo.reduce_pool(x, mask, mode)