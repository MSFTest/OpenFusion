# -*- coding: utf-8 -*-
"""MMDetection3D adapters for cached PointPainting features."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import torch


PFN_WEIGHT_KEY = "voxel_encoder.pfn_layers.0.linear.weight"


def append_cached_scores(
    points: torch.Tensor, cache_path: Union[str, Path]
) -> torch.Tensor:
    scores = np.load(str(cache_path), allow_pickle=False)
    if scores.dtype != np.uint8 or scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError(f"invalid semantic cache {cache_path}: {scores.shape} {scores.dtype}")
    if scores.shape[0] != points.shape[0]:
        raise ValueError(
            f"semantic cache row mismatch for {cache_path}: {scores.shape[0]} != {points.shape[0]}"
        )
    score_tensor = torch.from_numpy(scores.copy()).to(
        device=points.device, dtype=points.dtype
    ) / 255.0
    return torch.cat([points, score_tensor], dim=1)


def replace_points_tensor(points, tensor: torch.Tensor):
    """Rebuild a BasePoints-compatible object with the painted channel count."""
    return type(points)(
        tensor,
        points_dim=int(tensor.shape[1]),
        attribute_dims=points.attribute_dims,
    )


def expand_pfn_weight(source: torch.Tensor, semantic_channels: int = 4) -> torch.Tensor:
    """Insert zero semantic columns between raw inputs and PFN decorations."""
    semantic_channels = int(semantic_channels)
    if source.ndim != 2 or source.shape[1] != 10:
        raise ValueError(f"expected a (out_channels, 10) PFN weight, got {tuple(source.shape)}")
    expanded = source.new_zeros((source.shape[0], source.shape[1] + semantic_channels))
    expanded[:, :4] = source[:, :4]
    expanded[:, 4 + semantic_channels :] = source[:, 4:]
    return expanded


def expand_pointpillars_checkpoint(
    source_path: Union[str, Path], target_path: Union[str, Path]
) -> Path:
    source_path = Path(source_path)
    target_path = Path(target_path)
    checkpoint = torch.load(str(source_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if PFN_WEIGHT_KEY not in state_dict:
        raise KeyError(f"checkpoint has no {PFN_WEIGHT_KEY}")
    state_dict[PFN_WEIGHT_KEY] = expand_pfn_weight(state_dict[PFN_WEIGHT_KEY])
    metadata = checkpoint.setdefault("meta", {})
    metadata["pointpainting_input_channels"] = 8
    metadata["pointpainting_source_checkpoint"] = str(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target_path) + ".tmp")
    torch.save(checkpoint, str(temporary))
    os.replace(str(temporary), str(target_path))
    return target_path


try:
    from mmcv.transforms import BaseTransform
    from mmdet3d.registry import TRANSFORMS

    @TRANSFORMS.register_module()
    class AppendCachedPointSemantics(BaseTransform):
        """Append four cached semantic channels to an MMDetection3D points object."""

        def __init__(self, cache_dir: str):
            self.cache_dir = Path(cache_dir)

        def transform(self, results: dict) -> dict:
            lidar_path = results.get("lidar_path") or results.get("pts_filename")
            if lidar_path is None:
                lidar_path = results.get("lidar_points", {}).get("lidar_path")
            if lidar_path is None:
                raise KeyError("cannot determine lidar_path for semantic cache lookup")
            cache_path = self.cache_dir / f"{Path(lidar_path).stem}.npy"
            points = results["points"]
            painted = append_cached_scores(points.tensor, cache_path)
            results["points"] = replace_points_tensor(points, painted)
            return results

except ImportError:
    # Pure helpers remain importable in lightweight local test environments.
    AppendCachedPointSemantics = None
