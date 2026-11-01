"""Pure PyTorch pairwise IoU for rotated boxes in center-z convention."""

import torch


@torch.no_grad()
def box_iou3d_rotated(boxes1: torch.Tensor, boxes2: torch.Tensor, *, bottom_z: bool = False,
                      bev_only: bool = False) -> torch.Tensor:
    """Pairwise IoU for ``[x,y,z,dx,dy,dz,yaw]`` boxes, computed in fp32.

    ``bottom_z`` opts into bottom-centered z for legacy callers. The default is
    center-z. The result is detached, finite, clamped to [0, 1], and fp32.
    """
    if boxes1.ndim != 2 or boxes2.ndim != 2 or boxes1.shape[-1] < 7 or boxes2.shape[-1] < 7:
        raise ValueError("boxes must be [N, >=7] (x,y,z,dx,dy,dz,yaw,...)")
    if boxes1.device != boxes2.device:
        raise ValueError("boxes must be on the same device")
    n, m = boxes1.shape[0], boxes2.shape[0]
    if (n and (boxes1[:, 3:6] <= 0).any()) or (m and (boxes2[:, 3:6] <= 0).any()):
        raise ValueError("box dimensions dx, dy, and dz must be positive")
    if not n or not m:
        return torch.zeros((n, m), device=boxes1.device, dtype=torch.float32)
    a, b = boxes1[:, :7].float(), boxes2[:, :7].float()
    signs = a.new_tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])

    def corners(boxes):
        local = boxes[:, None, 3:5] * signs[None]
        c, s = boxes[:, 6].cos()[:, None], boxes[:, 6].sin()[:, None]
        x = local[..., 0] * c - local[..., 1] * s + boxes[:, None, 0]
        y = local[..., 0] * s + local[..., 1] * c + boxes[:, None, 1]
        return torch.stack((x, y), -1)

    ac, bc = corners(a), corners(b)
    acp, bcp = ac[:, None], bc[None]
    ae, be = acp[..., 1:, :] - acp[..., :-1, :], bcp[..., 1:, :] - bcp[..., :-1, :]
    ae = torch.cat((ae, acp[..., :1, :] - acp[..., -1:, :]), dim=-2)
    be = torch.cat((be, bcp[..., :1, :] - bcp[..., -1:, :]), dim=-2)

    def inside(points, boxes):
        delta = points - boxes[:, :, None, :2]
        c, s = boxes[:, :, 6].cos()[..., None], boxes[:, :, 6].sin()[..., None]
        lx = delta[..., 0] * c + delta[..., 1] * s
        ly = -delta[..., 0] * s + delta[..., 1] * c
        return ((lx.abs() <= boxes[:, :, None, 3] * .5 + 1e-6)
                & (ly.abs() <= boxes[:, :, None, 4] * .5 + 1e-6))

    a_inside_b = inside(acp.expand(n, m, 4, 2), b[None].expand(n, m, 7))
    b_inside_a = inside(bcp.expand(n, m, 4, 2), a[:, None].expand(n, m, 7))
    p, q = acp[..., :, None, :], bcp[..., None, :, :]
    r, s = ae[..., :, None, :], be[..., None, :, :]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    denominator = cross(r, s)
    safe_denominator = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
    t = cross(q - p, s) / safe_denominator
    u = cross(q - p, r) / safe_denominator
    valid_intersection = ((denominator.abs() > 1e-8) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))
    intersections = p + t[..., None] * r
    candidates = torch.cat((acp.expand(n, m, 4, 2), bcp.expand(n, m, 4, 2),
                            intersections.reshape(n, m, 16, 2)), dim=2)
    valid = torch.cat((a_inside_b, b_inside_a, valid_intersection.reshape(n, m, 16)), dim=2)
    counts = valid.sum(-1)
    centroid = (candidates * valid[..., None]).sum(-2) / counts.clamp_min(1)[..., None]
    angle = torch.atan2(candidates[..., 1] - centroid[..., None, 1],
                        candidates[..., 0] - centroid[..., None, 0])
    angle = torch.where(valid, angle, angle.new_full((), 10.))
    order = angle.argsort(-1)
    points = candidates.gather(2, order[..., None].expand(-1, -1, -1, 2))
    count = counts.clamp_min(1)
    row = torch.arange(24, device=boxes1.device).view(1, 1, 24)
    next_row = torch.where(row + 1 < count[..., None], row + 1, torch.zeros_like(row)).expand(n, m, -1)
    following = points.gather(2, next_row[..., None].expand(-1, -1, -1, 2))
    signed = points[..., 0] * following[..., 1] - points[..., 1] * following[..., 0]
    intersection_area = (signed * (row < counts[..., None])).sum(-1).abs() * .5
    area1, area2 = a[:, 3] * a[:, 4], b[:, 3] * b[:, 4]
    area_union = area1[:, None] + area2[None] - intersection_area
    if bev_only:
        iou = intersection_area / area_union.clamp_min(1e-6)
    else:
        if bottom_z:
            top_a, bottom_a = a[:, 2] + a[:, 5], a[:, 2]
            top_b, bottom_b = b[:, 2] + b[:, 5], b[:, 2]
        else:
            top_a, bottom_a = a[:, 2] + a[:, 5] / 2, a[:, 2] - a[:, 5] / 2
            top_b, bottom_b = b[:, 2] + b[:, 5] / 2, b[:, 2] - b[:, 5] / 2
        overlap_h = (torch.minimum(top_a[:, None], top_b[None])
                     - torch.maximum(bottom_a[:, None], bottom_b[None])).clamp_min(0)
        overlap = intersection_area * overlap_h
        volume1, volume2 = area1 * a[:, 5], area2 * b[:, 5]
        iou = overlap / (volume1[:, None] + volume2[None] - overlap).clamp_min(1e-6)
    return torch.nan_to_num(iou, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
