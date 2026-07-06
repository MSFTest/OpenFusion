from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from model.bevfusion.bevfusion_framework import BEVFusionFramework
from model.bevfusion.initialization.build_bevfusion_system import (
    BEVFusionPostProcessor,
    _freeze_module,
    _trainable_parameters,
)
from model.bevfusion.test_bevfusion import parse_native_ap40


class _Recorder(nn.Module):
    def __init__(self, name, calls, result):
        super().__init__()
        self.name = name
        self.calls = calls
        self.result = result

    def forward(self, *args, **kwargs):
        self.calls.append(self.name)
        return self.result


def test_framework_only_dispatches_modules_in_order():
    calls = []
    framework = BEVFusionFramework(
        camera_feature_module=_Recorder("camera", calls, torch.ones(1, 2, 2, 2)),
        camera_view_transform_module=_Recorder("view", calls, torch.ones(1, 2, 2, 2)),
        lidar_bev_module=_Recorder("lidar", calls, torch.ones(1, 2, 2, 2)),
        bev_fusion_module=_Recorder("fusion", calls, torch.ones(1, 2, 2, 2)),
        detection_task_module=_Recorder("detection", calls, {"scores": torch.ones(1)}),
    )

    output = framework({"images": torch.ones(1, 3, 8, 8)}, mode="test")

    assert calls == ["camera", "view", "lidar", "fusion", "detection"]
    assert output["scores"].item() == 1.0


def test_postprocessor_projects_nonzero_bbox_and_computes_alpha():
    post = BEVFusionPostProcessor(class_names=("Pedestrian", "Cyclist", "Car"))
    calib = {
        "P2": torch.tensor(
            [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
        "R0_rect": torch.eye(3),
        "Tr_velo2cam": torch.tensor(
            [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
    }
    prediction = {
        "boxes_3d": torch.tensor([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "scores_3d": torch.tensor([0.75]),
        "labels_3d": torch.tensor([2]),
    }

    annos = post([prediction], {"calibs": [calib], "image_shapes": [(80, 100)]})

    assert annos[0]["name"].tolist() == ["Car"]
    assert torch.as_tensor(annos[0]["bbox"]).abs().sum().item() > 0
    assert annos[0]["score"][0] == 0.75
    expected = annos[0]["rotation_y"][0] - math.atan2(
        annos[0]["location"][0, 0], annos[0]["location"][0, 2]
    )
    expected = (expected + math.pi) % (2 * math.pi) - math.pi
    assert abs(float(annos[0]["alpha"][0]) - expected) < 1e-5


def test_postprocessor_filters_boxes_outside_camera_view():
    post = BEVFusionPostProcessor(class_names=("Pedestrian", "Cyclist", "Car"))
    calib = {
        "P2": torch.tensor(
            [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
        "R0_rect": torch.eye(3),
        "Tr_velo2cam": torch.tensor(
            [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
    }
    prediction = {
        "boxes_3d": torch.tensor([[10.0, -100.0, 0.0, 4.0, 2.0, 1.5, 0.0]]),
        "scores_3d": torch.tensor([0.9]),
        "labels_3d": torch.tensor([2]),
    }

    annotation = post([prediction], {"calibs": [calib], "image_shapes": [(80, 100)]})[0]

    assert annotation["score"].size == 0


def test_frozen_parameters_are_excluded_from_optimizer_parameters():
    frozen = nn.Linear(4, 4)
    trainable = nn.Linear(4, 2)
    model = nn.ModuleDict({"frozen": frozen, "trainable": trainable})

    _freeze_module(frozen)
    params = list(_trainable_parameters(model))

    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 0
    assert sum(p.numel() for p in params) == 10


def test_parse_native_ap40_reads_three_difficulty_rows(tmp_path: Path):
    values = " ".join(["0.5"] * 41)
    for name in ("stats_car_detection.txt", "stats_car_detection_ground.txt"):
        (tmp_path / name).write_text(
            "\n".join([values, values, values]) + "\n", encoding="utf-8"
        )

    parsed = parse_native_ap40(tmp_path, class_names=("Car",))

    assert parsed["Car"]["3d_ap40"] == [50.0, 50.0, 50.0]
