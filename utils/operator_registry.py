# -*- coding: utf-8 -*-
from torch.library import Library


# 只允许一个 fusionrepo DEF
_lib = Library("fusionrepo", "DEF")


# ------------------------------------------------
# geometry / projection
# ------------------------------------------------

_lib.define(
    "points_to_image(Tensor points_3d, Tensor proj_matrix, float eps=1e-6) -> Tensor"
)

_lib.define(
    "compose_kitti_proj_matrix(Tensor P2, Tensor R0, Tensor Tr_velo_to_cam) -> Tensor"
)

_lib.define(
    "center_uv_to_bbox2d(Tensor uv, Tensor w, Tensor h) -> Tensor"
)

_lib.define(
    "corners_to_bbox2d(Tensor corners_pixels, Tensor image_shape) -> Tensor"
)

_lib.define(
    "box3d_to_corners(Tensor boxes_3d) -> Tensor"
)

_lib.define(
    "box3d_to_bev_box2d(Tensor boxes_3d, Tensor bev_extents) -> Tensor"
)

_lib.define(
    "batched_homogeneous_transform_and_normalize("
    "Tensor points, "
    "Tensor transform, "
    "bool normalize=False, "
    "float eps=1e-6"
    ") -> Tensor"
)

_lib.define(
    "camera_bev_geometry_construction("
    "Tensor bev_coords, "
    "Tensor intrinsics, "
    "Tensor extrinsics"
    ") -> Tensor"
)

_lib.define(
    "pixels_depth_to_camera_points("
    "Tensor pixels_2d, "
    "Tensor depths, "
    "Tensor K"
    ") -> Tensor"
)


# ------------------------------------------------
# feature sampling / transform
# ------------------------------------------------

_lib.define(
    "feature_sampling_by_grid(Tensor feature_map, Tensor grid) -> Tensor"
)

_lib.define(
    "feature_gather_by_pixel_index("
    "Tensor pixel_coords, "
    "Tensor feature_map, "
    "str rounding_mode='long', "
    "str layout='HWC', "
    "bool clamp_coords=True"
    ") -> Tensor"
)

_lib.define(
    "channel_reduce_1x1_2d(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor"
)

_lib.define(
    "feature_fusion_weighted_sum(Tensor features, Tensor? weights=None) -> Tensor"
)

_lib.define(
    "feature_fusion_broadcast_weighted_sum(Tensor features, Tensor? weights=None) -> Tensor"
)

_lib.define(
    "camera_lidar_bev_feature_fusion("
    "Tensor camera_bev_features, "
    "Tensor lidar_bev_features, "
    "Tensor attention_weights"
    ") -> Tensor"
)


# ------------------------------------------------
# IoU
# ------------------------------------------------

_lib.define(
    "iou2d_matrix(Tensor boxes_a, Tensor boxes_b, float eps=1e-6) -> Tensor"
)

_lib.define(
    "iou3d_matrix(Tensor boxes_a, Tensor boxes_b, float eps=1e-6) -> Tensor"
)

_lib.define(
    "anchor_assign_iou(Tensor anchors_3d, Tensor gt_boxes_3d, float pos_iou_thr, float neg_iou_thr) -> (Tensor, Tensor)"
)

_lib.define(
    "roi_assign_iou(Tensor proposals_3d, Tensor gt_boxes_3d, float pos_iou_thr, float neg_iou_thr) -> (Tensor, Tensor)"
)


# ------------------------------------------------
# NMS
# ------------------------------------------------

_lib.define(
    "nms3d(Tensor boxes_3d, Tensor scores, float iou_threshold) -> Tensor"
)

_lib.define(
    "nms_bev(Tensor boxes_bev, Tensor scores, float iou_threshold) -> Tensor"
)

_lib.define(
    "topk_after_nms(Tensor keep_indices, int top_k) -> Tensor"
)


# ------------------------------------------------
# pair / relation
# ------------------------------------------------

_lib.define(
    "scatter_pairs_to_matrix(Tensor pair_scores, Tensor pair_indices, int K, int N, float fill_value=-1e7, bool add_batch_dim=True) -> Tensor"
)

_lib.define(
    "pair_selector(Tensor iou_matrix, float iou_threshold=0.) -> Tensor"
)

_lib.define(
    "pair_feature_encoder(Tensor pair_indices, Tensor iou_matrix, Tensor boxes_3d, Tensor scores_2d, Tensor scores_3d) -> Tensor"
)


# ------------------------------------------------
# pooling
# ------------------------------------------------

_lib.define(
    "reduce_pool(Tensor x, int dim=1, str mode='max', bool keepdim=False) -> Tensor"
)

_lib.define(
    "generate_3d_anchor_grid(Tensor x_centers, Tensor z_centers, float y_center, Tensor anchor_sizes) -> Tensor"
)

_lib.define(
    "generate_roi_grid_points(Tensor rois, int grid_size) -> Tensor"
)

_lib.define(
    "gather_roi_features(Tensor rois, Tensor voxel_coords, Tensor voxel_features) -> (Tensor, Tensor)"
)

_lib.define(
    "pixel_to_grid_normalized(Tensor pixel_coords, int height, int width) -> Tensor"
)

_lib.define(
    "multihead_attention_fusion(Tensor query, Tensor key, Tensor value, int num_heads) -> Tensor"
)

_lib.define(
    "transformer_ffn_block(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2, Tensor ln_weight, Tensor ln_bias, float eps=1e-05) -> Tensor"
)

_lib.define(
"linear_projection(Tensor x, Tensor weight, Tensor bias) -> Tensor"
)


_lib.define(
    "compute_voxel_centroid(Tensor voxel_points, Tensor voxel_num_points) -> Tensor"
)

_lib.define(
    "feature_concatenation(Tensor[] features, int dim) -> Tensor"
)

_lib.define(
    "build_projection_matrix(Tensor cam_intrinsic, Tensor extrinsic) -> Tensor"
)

_lib.define(
    "append_camera_depth(Tensor points_3d, Tensor extrinsic, Tensor pixel_coords, float eps=1e-6) -> Tensor"
)

_lib.define(
    "generate_depth_bins(Tensor depth_range, int num_bins, int mode, float bin_size=0.0, float stride=0.0) -> Tensor"
)
_lib.define(
    "expand_depth_bins_to_image(Tensor depth_bins, int height, int width) -> Tensor"
)

_lib.define(
    "tokens_to_1x1_2d_map(Tensor x) -> Tensor"
)

_lib.define(
    "qkv_2d_map_to_tokens(Tensor query_map, Tensor key_map, Tensor value_map) -> Tensor[]"
)
_lib.define(
    "attention_tokens_to_batch(Tensor x) -> Tensor"
)

_lib.define(
    "prepare_trilinear_sampling_by_depth_slices("
    "Tensor frustum_features, "
    "Tensor frustum_coords, "
    "Tensor depth_bins, "
    "int height, "
    "int width"
    ") -> Tensor[]"
)

_lib.define(
    "fuse_depth_slice_samples("
    "Tensor sampled0, "
    "Tensor sampled1, "
    "Tensor depth_alpha"
    ") -> Tensor"
)

_lib.define(
    "image_frustum_feature_construction(Tensor image_features, Tensor depth_bins) -> Tensor"
)

_lib.define(
    "image_voxel_downsampling(Tensor image_voxel_features, int downsample_factor, int mode) -> Tensor"
)

_lib.define(
    "crop_points_in_3d_boxes("
    "Tensor rois, "
    "Tensor points_xyz, "
    "Tensor points_feat"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "score_threshold_filter("
    "Tensor boxes, "
    "Tensor scores, "
    "float threshold"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "gather_boxes_by_indices("
    "Tensor boxes, "
    "Tensor keep_indices"
    ") -> Tensor"
)
_lib.define(
    "generate_rois("
    "Tensor proposals, "
    "Tensor batch_indices"
    ") -> Tensor"
)

_lib.define(
    "pseudo_feature_encoding("
    "Tensor roi_pseudo_points, "
    "Tensor roi_indices, "
    "bool use_xyz=True"
    ") -> Tensor"
)


# ------------------------------------------------
# VirConv 新增通用算子
# ------------------------------------------------

_lib.define(
    "point_to_voxel_grouping("
    "Tensor points, "
    "Tensor voxel_size, "
    "Tensor point_cloud_range, "
    "int max_points_per_voxel, "
    "int max_voxels"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "voxel_stochastic_dropout("
    "Tensor voxel_coords, "
    "Tensor voxel_features, "
    "Tensor? bin_ids, "
    "float discard_rate, "
    "bool training"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "voxel_features_to_bev_scatter("
    "Tensor voxel_features, "
    "Tensor voxel_coords, "
    "int batch_size, "
    "int bev_h, "
    "int bev_w, "
    "str reduce='assign'"
    ") -> Tensor"
)


# ------------------------------------------------
# VirConv 几何/预处理算子（utils）
# ------------------------------------------------

_lib.define(
    "voxel_projection("
    "Tensor voxel_coords, "
    "Tensor voxel_size, "
    "Tensor point_cloud_range, "
    "Tensor proj_matrix"
    ") -> Tensor"
)

_lib.define(
    "distance_binning("
    "Tensor voxel_coords, "
    "Tensor voxel_size, "
    "Tensor point_cloud_range, "
    "Tensor bin_edges"
    ") -> Tensor"
)

_lib.define(
    "point_align(Tensor points) -> Tensor"
)


# ------------------------------------------------
# PointPainting 私有规范化/补全算子
# ------------------------------------------------

_lib.define(
    "pointpainting_lidar_to_camera_transform("
    "Tensor points_lidar, "
    "Tensor transform"
    ") -> Tensor"
)

_lib.define(
    "pointpainting_camera_to_image_projection("
    "Tensor points_camera, "
    "Tensor P, "
    "Tensor R0_rect, "
    "float eps=1e-6"
    ") -> Tensor"
)

_lib.define(
    "pointpainting_point_semantic_query("
    "Tensor pixel_coords, "
    "Tensor semantic_score_map"
    ") -> Tensor"
)

_lib.define(
    "pointpainting_point_feature_concatenation("
    "Tensor points_lidar, "
    "Tensor point_semantic_scores"
    ") -> Tensor"
)

_lib.define(
    "pointpainting_cyclist_semantic_refinement("
    "Tensor painted_points, "
    "int cyclist_index=-1, "
    "float threshold=0.3"
    ") -> Tensor"
)


# ------------------------------------------------
# CenterFusion 私有规范化算子
# ------------------------------------------------

_lib.define(
    "centerfusion_radar_pillar_expand("
    "Tensor radar_points, "
    "float pillar_height, "
    "float z_center=0.0"
    ") -> Tensor"
)

_lib.define(
    "centerfusion_prelim_3d_box("
    "Tensor centers_2d, "
    "Tensor depths, "
    "Tensor dims_3d, "
    "Tensor rots, "
    "Tensor K"
    ") -> Tensor"
)

_lib.define(
    "centerfusion_frustum_from_box3d("
    "Tensor box3d, "
    "Tensor K, "
    "float frustum_depth"
    ") -> Tensor"
)

_lib.define(
    "centerfusion_radar_frustum_association("
    "Tensor radar_pillars, "
    "Tensor frustums"
    ") -> Tensor"
)

_lib.define(
    "centerfusion_radar_feature_encoding("
    "Tensor radar_points, "
    "Tensor association"
    ") -> Tensor"
)

_lib.define(
    "centerfusion_radar_heatmap_generate("
    "Tensor radar_feats, "
    "Tensor centers_2d, "
    "Tensor feat_hw"
    ") -> Tensor"
)


# ------------------------------------------------
# F-ConvNet 几何/预处理与分组算子（utils）
# ------------------------------------------------

_lib.define(
    "frustum_points_to_image("
    "Tensor points_3d, "
    "Tensor K, "
    "float eps=1e-6"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "points_in_2d_boxes_mask("
    "Tensor points_2d, "
    "Tensor boxes_2d, "
    "int image_width, "
    "int image_height, "
    "bool use_image_bounds=True"
    ") -> Tensor"
)

_lib.define(
    "assign_values_to_bins("
    "Tensor values, "
    "Tensor bins, "
    "bool include_right=False, "
    "int invalid_index=-1"
    ") -> Tensor"
)

_lib.define(
    "build_sliding_depth_bins_and_assign("
    "Tensor depth, "
    "float bin_size, "
    "float stride, "
    "float z_min=-1.0, "
    "float z_max=-1.0"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "group_points_by_indices("
    "Tensor points, "
    "Tensor group_indices, "
    "int num_groups, "
    "int max_points_per_group, "
    "bool random_sample=False"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "normalize_grouped_points_with_centroid("
    "Tensor grouped_points, "
    "Tensor grouped_mask"
    ") -> Tensor"
)

_lib.define(
    "build_conv1d_feature_map(Tensor sequence_features) -> Tensor"
)


# ------------------------------------------------
# F-ConvNet 通用网络/融合函数式算子
# ------------------------------------------------

_lib.define(
    "masked_reduce_pool("
    "Tensor x, "
    "Tensor mask, "
    "int dim=1, "
    "str mode='max', "
    "bool keepdim=False"
    ") -> Tensor"
)

_lib.define(
    "align_and_concat_multiscale_1d(Tensor[] features) -> Tensor"
)


# ------------------------------------------------
# EPNeT 通用/私有规范化算子
# ------------------------------------------------

_lib.define(
    "feature_gate_multiply("
    "Tensor x, "
    "Tensor gate, "
    "bool apply_sigmoid=False"
    ") -> Tensor"
)

_lib.define(
    "projected_pixel_valid_mask("
    "Tensor uv, "
    "Tensor depth, "
    "int height, "
    "int width, "
    "float eps=0.0"
    ") -> Tensor"
)

_lib.define(
    "epnet_lidar_point_extractor(Tensor pointcloud) -> Tensor"
)

_lib.define(
    "epnet_lidar_to_image_projector("
    "Tensor lidar_points, "
    "Tensor R, "
    "Tensor t, "
    "Tensor K"
    ") -> Tensor[]"
)

_lib.define(
    "epnet_grid_generator("
    "int img_w, "
    "int img_h, "
    "Tensor uv, "
    "Tensor depth"
    ") -> Tensor[]"
)

_lib.define(
    "epnet_image_sampler("
    "Tensor feature_map, "
    "Tensor grid"
    ") -> Tensor"
)

_lib.define(
    "epnet_li_fusion_add("
    "Tensor Fp, "
    "Tensor Fi, "
    "bool apply_tanh=True"
    ") -> Tensor"
)

_lib.define(
    "epnet_li_fusion_muti("
    "Tensor x, "
    "Tensor y, "
    "bool sigmoid_y=False"
    ") -> Tensor"
)

_lib.define(
    "epnet_li_fusion_concat("
    "Tensor[] tensors, "
    "int dim=1"
    ") -> Tensor"
)


# ------------------------------------------------
# EPNet++ 私有规范化/补全算子
# ------------------------------------------------

_lib.define(
    "epnetpp_image_point_sampling("
    "Tensor Fi, "
    "Tensor xy"
    ") -> Tensor"
)

_lib.define(
    "epnetpp_point_to_grid_scatter("
    "Tensor point_features, "
    "Tensor point_indices, "
    "int grid_h, "
    "int grid_w"
    ") -> Tensor"
)

# ------------------------------------------------
# BEVfusion 私有规范化/补全算子
# ------------------------------------------------

_lib.define(
    "bevfusion_depth_weighted_feature_expand("
    "Tensor img_feats, "
    "Tensor depth_prob"
    ") -> Tensor"
)

_lib.define(
    "bevfusion_camera_geometry_transform("
    "Tensor points_3d, "
    "Tensor R, "
    "Tensor t"
    ") -> Tensor"
)

_lib.define(
    "bevfusion_camera_lift_2d_to_3d("
    "Tensor pixels_2d, "
    "Tensor depths, "
    "Tensor K, "
    "Tensor R, "
    "Tensor t"
    ") -> Tensor"
)

_lib.define(
    "bevfusion_lidar_voxelization("
    "Tensor points, "
    "Tensor voxel_size, "
    "Tensor point_cloud_range, "
    "int max_points_per_voxel, "
    "int max_voxels"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "bevfusion_lidar_flatten_z_to_bev("
    "Tensor feat_3d"
    ") -> Tensor"
)

_lib.define(
    "bevfusion_camera_bev_pooling("
    "Tensor feat_3d, "
    "Tensor bev_indices, "
    "int bev_h, "
    "int bev_w"
    ") -> Tensor"
)

_lib.define(
    "bevfusion_bev_concat_fusion("
    "Tensor bev_cam, "
    "Tensor bev_lidar, "
    "int dim=1"
    ") -> Tensor"
)


# ------------------------------------------------
# Faraway-Frustum 私有规范化与补全算子
# ------------------------------------------------

_lib.define(
    "faraway_mask_frustum_pointcloud_crop("
    "Tensor points_2d_img, Tensor points_3d_cam0, Tensor boxes_2d, "
    "Tensor masks_2d, int max_points=2048"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "faraway_frustum_histogram_clustering("
    "Tensor clusters, Tensor counts, int bins=20"
    ") -> Tensor"
)

_lib.define(
    "faraway_object_decision("
    "Tensor centroids, Tensor labels, Tensor depth_thresholds"
    ") -> Tensor"
)

_lib.define(
    "faraway_frustum_axis_alignment("
    "Tensor clusters, Tensor counts, Tensor frustum_angles"
    ") -> Tensor"
)

_lib.define(
    "faraway_centroid_coordinate_transform("
    "Tensor clusters, Tensor counts, Tensor centroids"
    ") -> Tensor"
)

_lib.define(
    "faraway_frustum_bev_projection("
    "Tensor clusters, Tensor counts, int bev_height, int bev_width, "
    "float x_range, float z_range"
    ") -> Tensor"
)

_lib.define(
    "faraway_assemble_3d_box("
    "Tensor centroids, Tensor regression_outputs"
    ") -> Tensor"
)
