import torch
from torch.library import Library


def compose_kitti_proj_matrix(R0: torch.Tensor,
                              Tr_velo2cam: torch.Tensor,
                              P2: torch.Tensor) -> torch.Tensor:
    device = P2.device
    dtype = P2.dtype

    # ----------------------------
    # R0: (3,3)/(3,4)/(4,4)
    # ----------------------------
    if R0.dim() != 2:
        raise ValueError(f"R0 must be 2D, got {tuple(R0.shape)}")

    if R0.shape == (3, 3):
        R0_33 = R0
    elif R0.shape == (3, 4):
        R0_33 = R0[:, :3]
    elif R0.shape == (4, 4):
        R0_33 = R0[:3, :3]
    else:
        raise ValueError(f"Unsupported R0 shape {tuple(R0.shape)}")

    R0_4 = torch.eye(4, device=device, dtype=dtype)
    R0_4[:3, :3] = R0_33.to(device=device, dtype=dtype)

    # ----------------------------
    # Tr_velo2cam: (3,4)/(4,4)/(3,3)
    # ----------------------------
    if Tr_velo2cam.dim() != 2:
        raise ValueError(f"Tr_velo2cam must be 2D, got {tuple(Tr_velo2cam.shape)}")

    if Tr_velo2cam.shape == (3, 4):
        Tr_34 = Tr_velo2cam
    elif Tr_velo2cam.shape == (4, 4):
        Tr_34 = Tr_velo2cam[:3, :]
    elif Tr_velo2cam.shape == (3, 3):
        Tr_34 = torch.zeros((3, 4), device=Tr_velo2cam.device, dtype=Tr_velo2cam.dtype)
        Tr_34[:, :3] = Tr_velo2cam
    else:
        raise ValueError(f"Unsupported Tr_velo2cam shape {tuple(Tr_velo2cam.shape)}")

    Tr_4 = torch.eye(4, device=device, dtype=dtype)
    Tr_4[:3, :4] = Tr_34.to(device=device, dtype=dtype)

    # ----------------------------
    # P2: (3,4)
    # ----------------------------
    P2_ = P2.to(device=device, dtype=dtype)

    proj_matrix = P2_ @ (R0_4 @ Tr_4)  # (3,4)

    return proj_matrix.clone()


_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_impl.impl("compose_kitti_proj_matrix", compose_kitti_proj_matrix)


def _compose_kitti_proj_matrix_meta(R0, Tr_velo2cam, P2):
    return torch.empty((3, 4), device="meta", dtype=P2.dtype)


_lib_meta = Library("fusionrepo", "IMPL", "Meta")
_lib_meta.impl("compose_kitti_proj_matrix", _compose_kitti_proj_matrix_meta)