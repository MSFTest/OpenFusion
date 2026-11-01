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

_lib.define(
    "project_points_to_normalized_image_grid("
    "Tensor points, "
    "Tensor transform, "
    "Tensor img_aug_matrix, "
    "Tensor lidar_aug_matrix, "
    "Tensor image_size, "
    "bool return_depth=False"
    ") -> Tensor[]"
)


# ------------------------------------------------
# feature sampling / transform
# ------------------------------------------------

_lib.define(
    "feature_sampling_by_grid(Tensor feature_map, Tensor grid) -> Tensor"
)

_lib.define(
    "sample_features_by_grid_v2("
    "Tensor feature_map, "
    "Tensor grid, "
    "Tensor? valid_mask=None, "
    "bool align_corners=False, "
    "str padding_mode='zeros'"
    ") -> Tensor"
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


# ------------------------------------------------
# IS-Fusion 通用格式算子与补全算子
# ------------------------------------------------

_lib.define(
    "isfusion_flatten_bev_grid("
    "Tensor bev_feats"
    ") -> (Tensor, Tensor)"
)

_lib.define(
    "isfusion_partition_bev_windows("
    "Tensor grid_features, "
    "Tensor grid_coords, "
    "int window_size, "
    "int shift_size"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "isfusion_pillar_image_feature_scatter("
    "Tensor sampled_features, "
    "Tensor pillar_coors, "
    "int bev_size"
    ") -> Tensor"
)

_lib.define(
    "isfusion_select_topk_instances("
    "Tensor ins_heatmap, "
    "Tensor bev_pos, "
    "int topk, "
    "int nms_kernel_size"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "isfusion_gather_instance_features("
    "Tensor scene_feats, "
    "Tensor topk_indices"
    ") -> Tensor"
)

_lib.define(
    "isfusion_reshape_scene_tokens_to_bev("
    "Tensor scene_tokens, "
    "int bev_h, "
    "int bev_w"
    ") -> Tensor"
)


# ------------------------------------------------
# BEVDet / DAL 私有格式统一算子
# ------------------------------------------------

_lib.define(
    "transform_points_to_ego(Tensor points, Tensor transform) -> Tensor"
)

_lib.define(
    "point_to_multiview_depth("
    "Tensor points, Tensor ego2img, Tensor post_rots, Tensor post_trans, "
    "Tensor image_size, float depth_min, float depth_max, int downsample"
    ") -> Tensor"
)

_lib.define(
    "create_lss_frustum(Tensor depth_cfg, Tensor input_size, int downsample) -> Tensor"
)

_lib.define(
    "project_frustum_to_ego("
    "Tensor frustum, Tensor sensor2ego, Tensor intrinsics, "
    "Tensor post_rots, Tensor post_trans, Tensor bda"
    ") -> Tensor"
)

_lib.define(
    "prepare_bev_pool_indices("
    "Tensor coor, Tensor grid_lower_bound, Tensor grid_interval, Tensor grid_size"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

_lib.define(
    "bev_pool_v2_pytorch("
    "Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, "
    "Tensor ranks_bev, Tensor bev_feat_shape, Tensor interval_starts, "
    "Tensor interval_lengths"
    ") -> Tensor"
)

_lib.define(
    "local_max_heatmap_nms(Tensor heatmap, int kernel_size, int dataset_id) -> Tensor"
)

_lib.define(
    "select_topk_proposals(Tensor heatmap, int num_proposals) "
    "-> (Tensor, Tensor, Tensor, Tensor)"
)

_lib.define(
    "sample_image_features_by_points("
    "Tensor ref_points, Tensor image_feats, Tensor sensor2ego, Tensor ego2global, "
    "Tensor intrinsics, Tensor post_rots, Tensor post_trans, Tensor bda, "
    "int img_feat_downsample"
    ") -> Tensor"
)

_lib.define(
    "compute_query_position(Tensor topk_indices, int bev_height, int bev_width) -> Tensor"
)

_lib.define(
    "build_reference_points("
    "Tensor center, Tensor height, Tensor voxel_size, Tensor pc_range, int out_size_factor"
    ") -> Tensor"
)


# ------------------------------------------------
# DeepInteraction 项目专属规范化算子
# ------------------------------------------------

_lib.define(
    "deepinteraction_image_to_lidar_sampling("
    "Tensor lidar_feature, Tensor image_feature, Tensor pillars, "
    "Tensor pillar_num_points, Tensor pillar_coordinates, Tensor lidar2image, "
    "Tensor lidar_augmentation, Tensor image_size"
    ") -> (Tensor, Tensor, Tensor, Tensor)"
)

_lib.define(
    "deepinteraction_lidar_to_image_sampling("
    "Tensor lidar_feature, Tensor image_feature_size, Tensor points, "
    "Tensor points_num, Tensor lidar2image, Tensor lidar_augmentation, "
    "Tensor image_size, Tensor point_cloud_range"
    ") -> Tensor"
)

_lib.define(
    "deepinteraction_decode_boxes("
    "Tensor center, Tensor height, Tensor dimension, Tensor rotation, "
    "Tensor velocity, Tensor point_cloud_range, Tensor voxel_size, "
    "int out_size_factor"
    ") -> Tensor"
)

_lib.define(
    "deepinteraction_lidar_box_corners("
    "Tensor boxes, float dimension_scale=1.0"
    ") -> Tensor"
)

_lib.define(
    "deepinteraction_image_roi_sampling("
    "Tensor image_feature, Tensor center, Tensor height, Tensor dimension, "
    "Tensor rotation, Tensor velocity, Tensor lidar2image, "
    "Tensor lidar_augmentation, Tensor image_size, Tensor image_shape, "
    "Tensor image_crop_offset, Tensor image_flip, Tensor point_cloud_range, "
    "Tensor voxel_size, int out_size_factor, int output_size, int sampling_ratio"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "deepinteraction_lidar_roi_sampling("
    "Tensor lidar_feature, Tensor center, Tensor height, Tensor dimension, "
    "Tensor rotation, Tensor velocity, Tensor point_cloud_range, "
    "Tensor voxel_size, int out_size_factor, int output_size, int sampling_ratio"
    ") -> Tensor"
)


# ------------------------------------------------
# CMT 项目专属规范化算子
# ------------------------------------------------

_lib.define(
    "cmt_bev_token_coordinates(Tensor bev_feature) -> Tensor"
)

_lib.define(
    "cmt_image_frustum_coordinates("
    "Tensor image_feature, Tensor img2lidar, Tensor pc_range, "
    "int depth_num, int pad_h, int pad_w"
    ") -> Tensor"
)

_lib.define(
    "cmt_anchor_multiview_projection("
    "Tensor reference_points, Tensor lidar2img, Tensor pc_range, "
    "int pad_h, int pad_w"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "cmt_projected_ray_coordinates("
    "Tensor projected_rays, Tensor img2lidar, Tensor pc_range, int depth_num"
    ") -> Tensor"
)


# ------------------------------------------------
# SparseFusion 算子与跨项目公共排序原语
# ------------------------------------------------

_lib.define(
    "descending_argsort_topk(Tensor scores, int k) -> (Tensor, Tensor)"
)

_lib.define(
    "sparsefusion_fusion_coordinate_grid("
    "Tensor valid_shapes, int bev_h, int bev_w, int[] image_heights, "
    "int[] image_widths, int image_stride=4"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)

_lib.define(
    "sparsefusion_bev_multiview_projection("
    "Tensor bev_pos, Tensor min_height, Tensor max_height, Tensor lidar2img, "
    "Tensor valid_shapes, int image_height, int image_width, "
    "float out_size_factor, float voxel_x, float voxel_y, "
    "float pc_min_x, float pc_min_y, float image_stride"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

_lib.define(
    "sparsefusion_camera_query_to_lidar("
    "Tensor center_2d, Tensor depth_2d, Tensor dim_2d, Tensor rot_2d, "
    "Tensor vel_2d, Tensor view_index, Tensor lidar2cam, Tensor cam_intrinsic, "
    "int image_height, int image_width, float pc_min_x, float pc_min_y, "
    "float pc_max_x, float pc_max_y, int bev_width, int bev_height"
    ") -> (Tensor, Tensor, Tensor)"
)

_lib.define(
    "sparsefusion_view_position_descriptor("
    "Tensor lidar_geometry, Tensor normal_image_position"
    ") -> Tensor"
)


# MSMDFusion 项目专属函数式算子。仅追加 schema。
_lib.define("msmdfusion_sparse_depth_map(Tensor real_pixels, Tensor offsets, int height, int width) -> Tensor")
_lib.define("msmdfusion_foreground_feature_sampling(Tensor feats, Tensor pixels, Tensor offsets, Tensor lidar2img, int input_width) -> (Tensor, Tensor)")
_lib.define("msmdfusion_modality_voxel_grouping(Tensor indices_3d, Tensor indices_2d, int batch_size) -> (Tensor, Tensor, Tensor, Tensor)")
_lib.define("msmdfusion_lidar_reference_assignment(Tensor indices_3d, Tensor indices_2d, Tensor features_2d, int fps_num, float radius, int max_cluster_samples, float dist_thresh) -> (Tensor, Tensor, Tensor)")
_lib.define("msmdfusion_sparse_height_flatten(Tensor features, Tensor indices, int batch_size, int depth, int height, int width) -> Tensor")


# UniTR 静态 shape 专属算子。仅追加 schema。
_lib.define("unitr_relative_window_coordinates(Tensor coors_in_win, int[] sparse_shape, int[] window_shape, bool normalize_pos=False) -> Tensor")
_lib.define("unitr_lidar_reference_points(Tensor coords_bzyx, float[] pc_range, int[] space_shape, int sample_num=1) -> Tensor")
_lib.define("unitr_image_token_lidar_coordinates(Tensor nearest_xyz, float[] pc_range, int[] space_shape) -> Tensor")
_lib.define("unitr_pseudo_lidar_grid(float[] pc_range, int[] space_shape, int sample_num, str device='cpu') -> Tensor")
# UniTR 动态 shape 专属算子。仅追加 schema。
_lib.define("unitr_sparse_window_set_partition(Tensor coords_bzyx, int[] sparse_shape, int[] window_shape, int[] shift0, int[] shift1, int set_size, int expand_max_voxels=1) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")
_lib.define("unitr_patch_nearest_projection(Tensor patch_coords, Tensor mapped_xyz, Tensor mapped_uv, Tensor counts, int[] hw_shape, int batch_size, int num_views) -> (Tensor, Tensor)")
_lib.define("unitr_lidar_to_image_projection(Tensor points, Tensor batch_idx, Tensor lidar2image, Tensor image_aug_matrix, Tensor? lidar2image_aug, int[] image_shape, int[] hw_shape, bool training, bool use_map, Tensor patch_coords, int num_views) -> (Tensor, Tensor)")
_lib.define("unitr_pseudo_grid_projection(Tensor points, Tensor lidar2image, Tensor image_aug_matrix, Tensor? lidar2image_aug, int[] image_shape, bool training, bool use_map) -> (Tensor, Tensor, Tensor)")


# TransFusion 算子落地：只追加 schema，不改变已有契约。
_lib.define('indexed_max_reduce(Tensor x, int dim=1, bool keepdim=False) -> (Tensor, Tensor)')
_lib.define('masked_prediction_fallback(Tensor updated, Tensor initial, Tensor processed) -> Tensor')
_lib.define('transfusion_feature_grid(Tensor anchor, int rows, int cols) -> Tensor')
_lib.define('transfusion_select_heatmap_proposals(Tensor lidar_logits, Tensor image_logits, int proposals=200, int kernel=3) -> (Tensor, Tensor, Tensor)')
_lib.define('transfusion_reverse_lidar_transform(Tensor points, Tensor rotation, float scale, Tensor translation, bool hflip, bool vflip, str[] flow) -> Tensor')
_lib.define('transfusion_project_proposals(Dict(str, Tensor) predictions, Tensor query_pos, Tensor[] lidar2img, Tensor[] scales, Tensor[] crops, Tensor[] image_shapes, Tensor[] padded_shapes, bool[] flips, Tensor[] rotations, float[] pcd_scales, Tensor[] translations, bool[] hflips, bool[] vflips, str[] flows, int out_size, float voxel_x, float voxel_y, float pc_x, float pc_y) -> (Tensor, Tensor, Tensor, Tensor)')
_lib.define('transfusion_smca_constraint(Tensor center_xy, Tensor corner_xy, Tensor visible, Tensor grid, int stride=4) -> (Tensor, Tensor)')
_lib.define('transfusion_aggregate_fusion_predictions(Tensor(a!)[] final, Tensor[] initial, Tensor visible, Tensor heatmap, Tensor indices, Tensor(b) dense) -> (Tensor(a!)[], Tensor, Tensor(b))')


# AutoAlignV2 项目专属算子；仅追加，公共计算核心由既有实现提供。
_lib.define('autoalignv2_recover_voxel_centers(Tensor coors, float[] voxel_size, float[] point_cloud_range) -> Tensor')
_lib.define('autoalignv2_restore_lidar_coordinates(Tensor centers, Tensor rotation, float scale, Tensor translation, bool horizontal, bool vertical, str[] flow) -> Tensor')
_lib.define('autoalignv2_project_voxels_to_image(Tensor centers, Tensor projection, Tensor scale, Tensor crop) -> Tensor')
_lib.define('autoalignv2_normalize_and_assign_camera(Tensor uvd, int height, int width, Tensor(a!) already) -> (Tensor, Tensor, Tensor, Tensor(a!))')
_lib.define('autoalignv2_sample_reference_features(Tensor[] features, Tensor grid, Tensor assigned) -> Tensor')
_lib.define('autoalignv2_initialize_cross_domain_query(Tensor points, Tensor reference, Tensor assigned) -> Tensor')
_lib.define('autoalignv2_deform_attention_backend(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_locations, Tensor attention_weights, int im2col_step=64, str backend="auto") -> Tensor')


# MVP 离线虚拟点生成：只追加专属 schema，不修改既有定义。
_lib.define('mvp_build_instance_semantics(Tensor[] masks, Tensor[] classes, Tensor[] scores, int height=900, int width=1600) -> (Tensor, Tensor)')
_lib.define('mvp_project_lidar_to_images(Tensor points, Tensor transforms, Tensor intrinsics, int height=900, int width=1600) -> Tensor')
_lib.define('mvp_construct_instance_frustums(Tensor masks, Tensor labels, Tensor projected, int height=900, int width=1600) -> (Tensor, Tensor, Tensor, Tensor)')
_lib.define('mvp_sample_instance_pixels(Tensor masks, int num_virtual=50, int height=900, int width=1600) -> (Tensor, Tensor)')
_lib.define('mvp_match_virtual_pixel_depths(Tensor(a!) virtual_points, Tensor virtual_cameras, Tensor valid, Tensor projected, float dist_thresh=3000.) -> (Tensor?, Tensor?, Tensor?, Tensor?)')
_lib.define('mvp_unproject_semantic_virtual_points(Tensor pixels, Tensor cameras, Tensor depths, Tensor indices, Tensor point_labels, Tensor intrinsics, Tensor transforms) -> Tensor')
_lib.define('mvp_aggregate_real_point_semantics(Tensor raw_points, Tensor point_labels, Tensor foreground, int num_camera=6) -> (Tensor, Tensor)')

_lib.define('mvp_bilinear_sample(Tensor im, Tensor x, Tensor y) -> Tensor')

_lib.define('mvp_rotate_points_along_z(Tensor points, Tensor angle) -> Tensor')

_lib.define('mvp_box_surface_centers(Tensor boxes) -> Tensor')

_lib.define('mvp_boxes_iou3d(Tensor boxes_a, Tensor boxes_b) -> Tensor')

_lib.define('mvp_rotated_nms(Tensor boxes, Tensor scores, float thresh, int pre_maxsize=1000, int post_max_size=83) -> Tensor')


# PointAugmenting：五个项目专属入口，只追加 schema；三个子功能复用已有公共原语。
_lib.define('pointaugmenting_transform_reference_points_to_cameras(Tensor points, Tensor ref_to_global, Tensor cameras_from_global) -> Tensor')
_lib.define('pointaugmenting_project_camera_points_to_images(Tensor camera_points, Tensor intrinsics) -> Tensor')
_lib.define('pointaugmenting_resolve_point_camera_correspondence(Tensor pixels, Tensor camera_points) -> Tensor')
_lib.define('pointaugmenting_normalize_sampling_coordinates(Tensor(a!) correspondence) -> Tensor(a!)')
_lib.define('pointaugmenting_sample_and_decorate_point_features(Tensor image_features, Tensor voxels, Tensor sample_grid, Tensor voxel_valid) -> Tensor')
