# -*- coding: utf-8 -*-
"""Faraway-Frustum 论文划分专用 PV-RCNN 训练配置。"""

import os

_base_ = [
    "mmdet3d::pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py",
]

data_root = os.environ.get(
    "FARAWAY_KITTI_DATA_ROOT",
    "model/faraway_frustum/data/kitti_paper/",
)
if not data_root.endswith("/"):
    data_root += "/"

class_names = ["Pedestrian", "Cyclist", "Car"]
metainfo = dict(classes=class_names)
backend_args = None
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
use_ground_plane = os.environ.get("FARAWAY_PVRCNN_USE_GROUND_PLANE", "0") == "1"
disable_db_sample = os.environ.get("FARAWAY_PVRCNN_DISABLE_DB_SAMPLE", "0") == "1"

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + "kitti_dbinfos_train.pkl",
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5),
    ),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    backend_args=backend_args,
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
]
if not disable_db_sample:
    train_pipeline.append(
        dict(
            type="ObjectSample",
            db_sampler=db_sampler,
            use_ground_plane=use_ground_plane,
        )
    )
train_pipeline.extend(
    [
        dict(type="RandomFlip3D", flip_ratio_bev_horizontal=0.5),
        dict(
            type="GlobalRotScaleTrans",
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05],
        ),
        dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
        dict(type="PointShuffle"),
        dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
    ]
)

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
        ],
    ),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

input_modality = dict(use_lidar=True, use_camera=False)
train_dataloader = dict(
    batch_size=int(os.environ.get("FARAWAY_PVRCNN_BATCH_SIZE", "2")),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=int(os.environ.get("FARAWAY_PVRCNN_REPEAT", "2")),
        dataset=dict(
            type="KittiDataset",
            data_root=data_root,
            ann_file="kitti_infos_train.pkl",
            data_prefix=dict(pts="training/velodyne"),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d="LiDAR",
            backend_args=backend_args,
        ),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="KittiDataset",
        data_root=data_root,
        data_prefix=dict(pts="training/velodyne"),
        ann_file="kitti_infos_val.pkl",
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="KittiMetric",
    ann_file=data_root + "kitti_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator

train_cfg = dict(
    _delete_=True,
    type="EpochBasedTrainLoop",
    max_epochs=int(os.environ.get("FARAWAY_PVRCNN_EPOCHS", "40")),
    val_begin=1,
    val_interval=int(os.environ.get("FARAWAY_PVRCNN_VAL_INTERVAL", "1")),
)
checkpoint_hook = dict(type="CheckpointHook", interval=1, max_keep_ckpts=3)
if os.environ.get("FARAWAY_PVRCNN_SAVE_BEST", "0") == "1":
    best_metric = (
        "Kitti metric/pred_instances_3d/KITTI/"
        "Car_3D_AP40_moderate_strict"
    )
    checkpoint_hook["save_best"] = best_metric
    checkpoint_hook["rule"] = "greater"

default_hooks = dict(
    checkpoint=checkpoint_hook,
    logger=dict(type="LoggerHook", interval=50),
)
randomness = dict(seed=3407, deterministic=False)
load_from = os.environ.get("FARAWAY_PVRCNN_INIT_CHECKPOINT") or None
resume = False

if os.environ.get("FARAWAY_PVRCNN_FINETUNE", "0") == "1":
    finetune_lr = float(os.environ.get("FARAWAY_PVRCNN_LR", "0.0001"))
    max_epochs = int(os.environ.get("FARAWAY_PVRCNN_EPOCHS", "5"))
    optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(
            type="AdamW",
            lr=finetune_lr,
            betas=(0.95, 0.99),
            weight_decay=0.01,
        ),
        clip_grad=dict(max_norm=10, norm_type=2),
    )
    param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=0.1,
            begin=0,
            end=min(500, max(1, 3724 // max(train_dataloader["batch_size"], 1))),
            by_epoch=False,
        ),
        dict(
            type="CosineAnnealingLR",
            T_max=max_epochs,
            eta_min=finetune_lr * 0.05,
            begin=0,
            end=max_epochs,
            by_epoch=True,
        ),
    ]
