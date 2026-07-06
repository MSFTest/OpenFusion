# -*- coding: utf-8 -*-
import torch
from torch.library import Library

_lib_impl = Library("fusionrepo", "IMPL", "CompositeExplicitAutograd")
_lib_meta = Library("fusionrepo", "IMPL", "Meta")


def _normalize_pair_scores(pair_scores: torch.Tensor) -> torch.Tensor:
    if pair_scores.dim() == 3:
        pair_scores = pair_scores.squeeze()

    if pair_scores.dim() == 2:
        if pair_scores.shape[1] == 1:
            pair_scores = pair_scores[:, 0]
        elif pair_scores.shape[1] == 2:
            if pair_scores.numel() == 0:
                pair_scores = pair_scores.new_empty((0,))
            else:
                pair_scores = torch.softmax(pair_scores, dim=1)[:, 1]
        else:
            raise ValueError(f"pair_scores must be (p,), (p,1) or (p,2), got {tuple(pair_scores.shape)}")

    if pair_scores.dim() != 1:
        raise ValueError(f"pair_scores must be (p,) or (1,1,p) or (p,1)/(p,2), got {tuple(pair_scores.shape)}")

    return pair_scores


def _coerce_pair_indices(pair_indices: torch.Tensor) -> torch.Tensor:
    if pair_indices.dim() == 1:
        if pair_indices.numel() == 0:
            return pair_indices.reshape(0, 2)
        if pair_indices.numel() == 2:
            return pair_indices.reshape(1, 2)
    return pair_indices


def _scatter_pairs_to_matrix_impl(pair_scores: torch.Tensor,
                                 pair_indices: torch.Tensor,
                                 K: int,
                                 N: int,
                                 fill_value: float = -1e7,
                                 add_batch_dim: bool = True) -> torch.Tensor:
    pair_scores = _normalize_pair_scores(pair_scores)

    device = pair_scores.device
    dtype = pair_scores.dtype

    if add_batch_dim:
        matrix = torch.full((1, int(K), int(N)), float(fill_value), device=device, dtype=dtype)
    else:
        matrix = torch.full((int(K), int(N)), float(fill_value), device=device, dtype=dtype)

    p = int(pair_scores.shape[0])
    if p == 0:
        return matrix

    pair_indices = _coerce_pair_indices(pair_indices)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError(f"pair_indices must be (p,2), got {tuple(pair_indices.shape)}")
    if int(pair_indices.shape[0]) != p:
        raise ValueError(f"pair_scores and pair_indices must have same length, got {p} and {int(pair_indices.shape[0])}")

    i_idx = pair_indices[:, 0].long()
    j_idx = pair_indices[:, 1].long()

    if add_batch_dim:
        matrix[0, i_idx, j_idx] = pair_scores
    else:
        matrix[i_idx, j_idx] = pair_scores
    return matrix


def _scatter_pairs_to_matrix_meta(pair_scores: torch.Tensor,
                                 pair_indices: torch.Tensor,
                                 K: int,
                                 N: int,
                                 fill_value: float = -1e7,
                                 add_batch_dim: bool = True) -> torch.Tensor:
    if add_batch_dim:
        return torch.empty((1, int(K), int(N)), device="meta", dtype=pair_scores.dtype)
    return torch.empty((int(K), int(N)), device="meta", dtype=pair_scores.dtype)


_lib_impl.impl("scatter_pairs_to_matrix", _scatter_pairs_to_matrix_impl)
_lib_meta.impl("scatter_pairs_to_matrix", _scatter_pairs_to_matrix_meta)


def scatter_pairs_to_matrix(pair_scores: torch.Tensor,
                            pair_indices: torch.Tensor,
                            K: int,
                            N: int,
                            fill_value: float = -1e7,
                            add_batch_dim: bool = True) -> torch.Tensor:
    return torch.ops.fusionrepo.scatter_pairs_to_matrix(
        pair_scores, pair_indices, int(K), int(N), float(fill_value), bool(add_batch_dim)
    )